import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import itertools
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join


def _torch_load_compat(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


def _flag_enabled(data, key, default=False):
    value = data.get(key, default)
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, (list, tuple)):
        return any(bool(v) for v in value)
    return bool(value)


# =========================================================
# 课设优化模块：物理与结构约束损失函数
# =========================================================

class LaplacianEdgeLoss(nn.Module):
    """用固定拉普拉斯核提取二阶边缘，约束预测与 GT 的高频细节一致。
    灵感来源：Dong et al. (ICCV 2021, Location-Aware SIRR) 将拉普拉斯核用于网络特征提取，
    此处将其适配为独立的损失函数。"""
    def __init__(self, device='cpu'):
        super(LaplacianEdgeLoss, self).__init__()
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = kernel.repeat(3, 1, 1, 1).to(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_T, target_T):
        pred_lap = F.conv2d(pred_T, self.kernel, padding=1, groups=3)
        target_lap = F.conv2d(target_T, self.kernel, padding=1, groups=3)
        return self.l1_loss(pred_lap, target_lap)


class GradientExclusionLoss(nn.Module):
    """梯度互斥损失：惩罚透射层与伪反射层的梯度重叠，迫使结构分离。
    来源：Zhang et al. 提出，DSRNet (Hu et al., ICCV 2023) 沿用 (Eq. 6)。
    此处针对单流架构，使用 R_pseudo = I - T_hat 替代双流预测。"""
    def __init__(self):
        super(GradientExclusionLoss, self).__init__()

    def forward(self, pred_T, I):
        R_pseudo = torch.clamp(I - pred_T, 0.0, 1.0)
        grad_T_x = pred_T[:, :, :, 1:] - pred_T[:, :, :, :-1]
        grad_T_y = pred_T[:, :, 1:, :] - pred_T[:, :, :-1, :]
        grad_R_x = R_pseudo[:, :, :, 1:] - R_pseudo[:, :, :, :-1]
        grad_R_y = R_pseudo[:, :, 1:, :] - R_pseudo[:, :, :-1, :]
        loss_x = torch.mean(torch.abs(grad_T_x * grad_R_x))
        loss_y = torch.mean(torch.abs(grad_T_y * grad_R_y))
        return loss_x + loss_y


class MaxRFMaskWeightedLoss(nn.Module):
    """MaxRF 掩膜加权 L1 损失：对强反射区域施加更高的重建惩罚。
    MaxRF 公式来自 RRW (arXiv 2023, Revisiting SIRR In the Wild)；
    将其作为损失权重的用法是本方案的原创适配。"""
    def __init__(self, weight_boost=1.5):
        super(MaxRFMaskWeightedLoss, self).__init__()
        self.weight_boost = weight_boost

    def forward(self, pred_T, target_T, I):
        # x 方向梯度 (N,C,H,W-1)，右边补一列 0 → (N,C,H,W)
        grad_I_x = F.pad(torch.abs(I[:, :, :, 1:] - I[:, :, :, :-1]), (0, 1, 0, 0))
        grad_T_x = F.pad(torch.abs(target_T[:, :, :, 1:] - target_T[:, :, :, :-1]), (0, 1, 0, 0))
        # y 方向梯度 (N,C,H-1,W)，下方补一行 0 → (N,C,H,W)
        grad_I_y = F.pad(torch.abs(I[:, :, 1:, :] - I[:, :, :-1, :]), (0, 0, 0, 1))
        grad_T_y = F.pad(torch.abs(target_T[:, :, 1:, :] - target_T[:, :, :-1, :]), (0, 0, 0, 1))
        # 现在形状一致，可以相加
        grad_I = grad_I_x + grad_I_y
        grad_T_gt = grad_T_x + grad_T_y
        mask = (grad_I > grad_T_gt).float()
        l1_diff = torch.abs(pred_T - target_T)
        weighted_loss = l1_diff * (1.0 + (self.weight_boost - 1.0) * mask)
        return torch.mean(weighted_loss)


class FeatureDecorrelationLoss(nn.Module):
    """特征去相关损失：约束 T_hat 与伪反射层在 VGG 特征空间中的独立性。
    灵感来源：DAD (Zou et al., CVPR 2020) 的 Separation-Critic 思想，
    此处用余弦相似度作为其轻量级替代。"""
    def __init__(self, vgg, layer_idx=21):
        super(FeatureDecorrelationLoss, self).__init__()
        self.vgg = vgg
        self.layer_idx = layer_idx
        device = next(vgg.parameters()).device
        self.normalize = losses.MeanShift(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)

    def forward(self, pred_T, I):
        R_pseudo = torch.clamp(I - pred_T, 0.0, 1.0)
        T_norm = self.normalize(pred_T)
        R_norm = self.normalize(R_pseudo)
        T_feat = self.vgg(T_norm, [self.layer_idx])[0]
        R_feat = self.vgg(R_norm, [self.layer_idx])[0]
        T_flat = T_feat.view(T_feat.size(0), -1)
        R_flat = R_feat.view(R_feat.size(0), -1)
        T_normed = F.normalize(T_flat, dim=1)
        R_normed = F.normalize(R_flat, dim=1)
        cos_sim = (T_normed * R_normed).sum(dim=1)
        return cos_sim.abs().mean()


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        
        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class ERRNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])                
        
        self.input = input
        
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.data_name = data_name

        self.issyn = not _flag_enabled(data, 'real', default=False)
        self.aligned = not _flag_enabled(data, 'unaligned', default=False)
        
        if target_t is not None:            
            self.target_edge = self.edge_map(self.target_t)         
            
    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            self.forward()

            output_i = tensor2im(self.output_i)
            target = tensor2im(self.target_t)

            if self.aligned:
                h = min(output_i.shape[0], target.shape[0])
                w = min(output_i.shape[1], target.shape[1])
                res = index.quality_assess(output_i[:h, :w], target[:h, :w])
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))
                    if suffix is not None:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'{}_{}.png'.format(self.opt.name, suffix)))
                    else:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 'transmission_layer', str(self._count)+'.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'blended', str(self._count)+'.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                return 
        
        with torch.no_grad():
            output_i = self.forward()
            output_i = tensor2im(output_i)
                # if os.path.exists(join(savedir, name,'t_output.png')):
                #     i = 2
                #     while True:
                #         if not os.path.exists(join(savedir, name,'t_output_{}.png'.format(i))):
                #             Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'t_output_{}.png'.format(i)))
                #             break
                #         i += 1
                # else:
                #     Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'t_output.png'))
            if self.data_name is not None and savedir is not None:                
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class ERRNetModel(ERRNetBase):
    def name(self):
        return 'errnet'
        
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device("cuda:%d" % self.gpu_ids[0] if len(self.gpu_ids) > 0 else "cpu")

        in_channels = 3
        self.vgg = None
        
        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        
        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        networks.init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss

            # [新增] 课设优化损失函数实例化
            self.laplacian_loss_fn = LaplacianEdgeLoss(device=self.device)
            self.exclusion_loss_fn = GradientExclusionLoss()
            self.maxrf_loss_fn = MaxRFMaskWeightedLoss(weight_boost=1.5)
            # FeatureDecorrelationLoss 依赖 VGG，仅在 --hyper 启用时创建
            self.fea_decorr_loss_fn = None
            if self.vgg is not None:
                self.fea_decorr_loss_fn = FeatureDecorrelationLoss(self.vgg, layer_idx=21)

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(), 
                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None
        # [新增] 初始化课设损失变量
        self.loss_maxrf = None
        self.loss_exclusion = None
        self.loss_laplacian = None
        self.loss_fea_decorr = None
        self.loss_vgg_ms = None
        self.loss_gradient = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t)
            self.loss_G += self.loss_G_GAN * self.opt.lambda_gan
        
        if self.aligned:
            # 基础像素损失（MSE + Gradient），权重由 lambda_pixel 动态控制
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            self.loss_G += self.loss_icnn_pixel * self.opt.lambda_pixel
            
            # VGG 感知损失
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)
            self.loss_G += self.loss_icnn_vgg * self.opt.lambda_vgg

            # [v3 新增] 多尺度感知损失：在 1/2 和 1/4 分辨率上额外计算 VGG 损失
            # 来源：IBCLN (Li et al., CVPR 2020) 的多尺度感知损失策略
            if getattr(self.opt, 'lambda_vgg_ms', 0) > 0:
                self.loss_vgg_ms = 0
                for scale in [0.5, 0.25]:
                    pred_scaled = F.interpolate(self.output_i, scale_factor=scale,
                        mode='bilinear', align_corners=False)
                    target_scaled = F.interpolate(self.target_t, scale_factor=scale,
                        mode='bilinear', align_corners=False)
                    self.loss_vgg_ms += self.loss_dic['t_vgg'].get_loss(pred_scaled, target_scaled)
                self.loss_vgg_ms /= 2.0
                self.loss_G += self.loss_vgg_ms * self.opt.lambda_vgg_ms

            # [v3 新增] 独立梯度惩罚：从 pixel loss 中解耦，可独立调节权重
            # 来源：ToT (NeurIPS 2021) 重建损失中的梯度场约束
            if getattr(self.opt, 'lambda_gradient', 0) > 0:
                pred_grad_x, pred_grad_y = losses.compute_gradient(self.output_i)
                target_grad_x, target_grad_y = losses.compute_gradient(self.target_t)
                self.loss_gradient = F.l1_loss(pred_grad_x, target_grad_x) + \
                                     F.l1_loss(pred_grad_y, target_grad_y)
                self.loss_G += self.loss_gradient * self.opt.lambda_gradient

            # [新增] MaxRF 掩膜加权损失
            if getattr(self.opt, 'lambda_maxrf', 0) > 0:
                self.loss_maxrf = self.maxrf_loss_fn(
                    self.output_i, self.target_t, self.input)
                self.loss_G += self.loss_maxrf * self.opt.lambda_maxrf

            # [新增] 梯度互斥损失
            if getattr(self.opt, 'lambda_exclusion', 0) > 0:
                self.loss_exclusion = self.exclusion_loss_fn(
                    self.output_i, self.input)
                self.loss_G += self.loss_exclusion * self.opt.lambda_exclusion

            # [新增] 拉普拉斯边缘损失
            if getattr(self.opt, 'lambda_laplacian', 0) > 0:
                self.loss_laplacian = self.laplacian_loss_fn(
                    self.output_i, self.target_t)
                self.loss_G += self.loss_laplacian * self.opt.lambda_laplacian

            # [新增] 特征去相关损失（复用 VGG，仅多一次 conv4_2 提取）
            if getattr(self.opt, 'lambda_fea_decorr', 0) > 0 and self.fea_decorr_loss_fn is not None:
                self.loss_fea_decorr = self.fea_decorr_loss_fn(
                    self.output_i, self.input)
                self.loss_G += self.loss_fea_decorr * self.opt.lambda_fea_decorr
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        # without edge
        input_i = self.input

        if self.vgg is not None:
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
            input_i = [input_i]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)

        output_i = self.net_i(input_i)

        self.output_i = output_i

        return output_i
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
            
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        # [新增] 课设损失项监控
        if self.loss_maxrf is not None:
            ret_errors['MaxRF'] = self.loss_maxrf.item()
        if self.loss_exclusion is not None:
            ret_errors['Exclusion'] = self.loss_exclusion.item()
        if self.loss_laplacian is not None:
            ret_errors['Laplacian'] = self.loss_laplacian.item()
        if self.loss_fea_decorr is not None:
            ret_errors['FeaDecorr'] = self.loss_fea_decorr.item()
        if self.loss_vgg_ms is not None:
            ret_errors['VGG_MS'] = self.loss_vgg_ms.item()
        if self.loss_gradient is not None:
            ret_errors['Gradient'] = self.loss_gradient.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals       

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if icnn_path is None:
            model_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)
            state_dict = _torch_load_compat(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model.net_i.load_state_dict(state_dict['icnn'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = _torch_load_compat(icnn_path, map_location=torch.device('cpu'))
            model.net_i.load_state_dict(state_dict['icnn'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            # if model.isTrain:
            #     model.optimizer_G.load_state_dict(state_dict['opt_g'])

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])
            
        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict


class NetworkWrapper(ERRNetBase):
    # You can use this class to wrap other module into our training framework (\eg BDN module)
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")

    def print_network(self):
        print('--------------------- NetworkWrapper ---------------------')
        networks.print_network(self.net)

    def _eval(self):
        self.net.eval()

    def _train(self):
        self.net.train()

    def initialize(self, opt, net):
        BaseModel.initialize(self, opt)
        self.device = torch.device("cuda:%d" % self.gpu_ids[0] if len(self.gpu_ids) > 0 else "cpu")
        self.net = net.to(self.device)
        self.edge_map = EdgeMap(scale=1).to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[31]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
                
            else:
                raise NotImplementedError            
            
            self.loss_dic['t_cx'] = cxloss

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net.parameters(), 
                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

            # define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
            self._init_optimizer([self.optimizer_D])
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)
        
    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False
                    
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
                
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            # self.loss_G += self.loss_icnn_pixel
            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
            # self.loss_G += self.loss_fm * self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        raise NotImplementedError
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()
        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)
        return ret_visuals

    def state_dict(self):
        state_dict = self.net.state_dict()
        return state_dict
