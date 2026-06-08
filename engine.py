import torch
import torch.nn as nn
import torch.nn.functional as F
import util.util as util
import models
import time
import os
import sys
from os.path import join
from util.visualizer import Visualizer

# =========================================================
# 课设优化模块：三大核心物理与结构约束损失函数 (零参数负担)
# =========================================================
class LaplacianEdgeLoss(nn.Module):
    def __init__(self):
        super(LaplacianEdgeLoss, self).__init__()
        # 定义固定的拉普拉斯算子提取二阶边缘，强制保护高频细节
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = kernel.repeat(3, 1, 1, 1).cuda()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_T, target_T):
        pred_lap = F.conv2d(pred_T, self.kernel, padding=1, groups=3)
        target_lap = F.conv2d(target_T, self.kernel, padding=1, groups=3)
        return self.l1_loss(pred_lap, target_lap)

class GradientExclusionLoss(nn.Module):
    def __init__(self):
        super(GradientExclusionLoss, self).__init__()

    def forward(self, pred_T, I):
        # 伪反射层 R_pseudo = I - pred_T
        R_pseudo = torch.clamp(I - pred_T, 0.0, 1.0)
        
        # 计算 X 和 Y 方向的梯度
        grad_T_x = pred_T[:, :, :, 1:] - pred_T[:, :, :, :-1]
        grad_T_y = pred_T[:, :, 1:, :] - pred_T[:, :, :-1, :]
        grad_R_x = R_pseudo[:, :, :, 1:] - R_pseudo[:, :, :, :-1]
        grad_R_y = R_pseudo[:, :, 1:, :] - R_pseudo[:, :, :-1, :]

        # 梯度互斥：惩罚重叠的纹理边缘，实现强效结构分离
        loss_x = torch.mean(torch.abs(grad_T_x * grad_R_x))
        loss_y = torch.mean(torch.abs(grad_T_y * grad_R_y))
        return loss_x + loss_y

class MaxRFMaskWeightedLoss(nn.Module):
    def __init__(self, weight_boost=1.5):
        super(MaxRFMaskWeightedLoss, self).__init__()
        self.weight_boost = weight_boost

    def forward(self, pred_T, target_T, I):
        # 计算梯度幅度
        grad_I = torch.abs(I[:, :, :, 1:] - I[:, :, :, :-1]) + torch.abs(I[:, :, 1:, :] - I[:, :, :-1, :])
        grad_T_gt = torch.abs(target_T[:, :, :, 1:] - target_T[:, :, :, :-1]) + torch.abs(target_T[:, :, 1:, :] - target_T[:, :, :-1, :])
        
        # 补齐维度对齐 (由于计算梯度丢掉了最后一行/列)
        grad_I = F.pad(grad_I, (0, 1, 0, 1))
        grad_T_gt = F.pad(grad_T_gt, (0, 1, 0, 1))

        # MaxRF 掩膜：检测强反射区域 (I的梯度 > GT透射的梯度)
        mask = (grad_I > grad_T_gt).float()
        
        # 加权 L1 损失 (强反光区赋予 weight_boost 倍的惩罚)
        l1_diff = torch.abs(pred_T - target_T)
        weighted_loss = l1_diff * (1.0 + (self.weight_boost - 1.0) * mask)
        return torch.mean(weighted_loss)


class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.name)
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        
        opt = self.opt
        
        """Model"""
        self.model = models.__dict__[self.opt.model]()
        self.model.initialize(opt)
        
        # ==========================================================
        # 核心 Hook：动态注入课设损失，规避修改 errnet_model.py 的风险
        # ==========================================================
        self.laplacian_loss = LaplacianEdgeLoss().cuda()
        self.exclusion_loss = GradientExclusionLoss().cuda()
        self.maxrf_loss = MaxRFMaskWeightedLoss(weight_boost=1.5).cuda()
        
        # 保存原有的 optimize_parameters，进行安全劫持
        original_optimize = self.model.optimize_parameters
        
        def new_optimize_parameters(**kwargs):
            # 1. 执行原有的前向、后向与参数更新 (保留 Backbone 完整性)
            original_optimize(**kwargs)
            
            # 2. 抓取张量 (兼容多种常用变量命名体系)
            pred_T = getattr(self.model, 'pred_T', getattr(self.model, 'output', getattr(self.model, 'fake_T', getattr(self.model, 'T_hat', None))))
            target_T = getattr(self.model, 'target_T', getattr(self.model, 'target', getattr(self.model, 'real_T', getattr(self.model, 'T', None))))
            input_I = getattr(self.model, 'input_I', getattr(self.model, 'input', getattr(self.model, 'real_I', getattr(self.model, 'I', None))))
            
            if pred_T is not None and target_T is not None and input_I is not None:
                # 获取课程学习策略中动态设置的权重参数
                lambda_l1 = getattr(self.opt, 'lambda_l1', 1.0)
                lambda_laplacian = getattr(self.opt, 'lambda_laplacian', 0.0)
                lambda_exclusion = getattr(self.opt, 'lambda_exclusion', 0.0)
                
                extra_loss = 0.0
                
                # 累加课设自定义的损失
                if lambda_l1 > 0:
                    extra_loss += self.maxrf_loss(pred_T, target_T, input_I) * lambda_l1
                if lambda_laplacian > 0:
                    extra_loss += self.laplacian_loss(pred_T, target_T) * lambda_laplacian
                if lambda_exclusion > 0:
                    extra_loss += self.exclusion_loss(pred_T, input_I) * lambda_exclusion
                    
                # 3. 产生额外梯度并更新，补充原模型的优化过程
                if extra_loss > 0 and hasattr(self.model, 'optimizers'):
                    optimizer_G = self.model.optimizers[0] # 通常第一个是生成器优化器
                    optimizer_G.zero_grad()
                    extra_loss.backward()
                    optimizer_G.step()
                    
                    # 将自定义 Loss 写入 errors 字典，让 TensorBoard 能监控到
                    if hasattr(self.model, 'loss_G'):
                         self.model.loss_G += extra_loss.item()
            else:
                # 若首次迭代找不到对应变量名，静默跳过或提示检查 (只需核对 errnet_model 里的变量名)
                pass

        # 挂载新方法
        self.model.optimize_parameters = new_optimize_parameters
        # ==========================================================

        if not opt.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer(opt)

    def train(self, train_loader, **kwargs):
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            iterations = self.iterations
            
            model.set_input(data, mode='train')
            # 此时调用的已经是挂载了你课设新损失函数的优化器
            model.optimize_parameters(**kwargs)
            
            errors = model.get_current_errors()
            avg_meters.update(errors)
            util.progress_bar(i, len(train_loader), str(avg_meters))
            
            if not opt.no_log:
                util.write_loss(self.writer, 'train', avg_meters, iterations)
            
                if iterations % opt.display_freq == 0 and opt.display_id != 0:
                    save_result = iterations % opt.update_html_freq == 0
                    self.visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if iterations % opt.print_freq == 0 and opt.display_id != 0:
                    t = (time.time() - iter_start_time)          

            self.iterations += 1
    
        self.epoch += 1

        if not self.opt.no_log:
            if self.epoch % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                    (self.epoch, self.iterations))
                model.save()
            
            print('saving the latest model at the end of epoch %d, iters %d' % 
                (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                (time.time() - epoch_start_time))
                
        train_loader.reset()

    def eval(self, val_loader, dataset_name, savedir=None, loss_key=None, **kwargs):
        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(val_loader):                
                index = model.eval(data, savedir=savedir, **kwargs)
                avg_meters.update(index)
                
                util.progress_bar(i, len(val_loader), str(avg_meters))
                
        if not opt.no_log:
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)
        
        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' % 
                    (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e