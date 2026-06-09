from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

datadir = './datasets/processed_data'
datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True, 
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)
train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])
train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=True)

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))
eval_dataset_real = datasets.CEILTestDataset(join(datadir, 'real20'), size=20, max_long_edge=512)

eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

"""Main Loop"""
engine = Engine(opt)

def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

if opt.resume:
    res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')

# =========================================================
# 课程学习策略 v3（对抗真实场景过拟合）
#
# v2 问题诊断：
#   Stage 3 在合成数据 70% 比例下激活 Laplacian + FeatureDecorrelation，
#   导致模型学到合成数据特有的纹理套路，真实场景四数据集同步退化。
#
# v3 修复方案（三板斧）：
#   1. Stage 3 翻转 fusion ratio → [0.3, 0.7]，让模型在精细调优阶段
#      看到 70% 真实数据，用数据分布对抗损失函数带来的过拟合倾向
#   2. 引入多尺度感知损失（IBCLN, Li et al. CVPR 2020）：
#      在 1/2 和 1/4 分辨率上额外计算 VGG 感知损失，
#      提供尺度不变的结构监督，改善真实场景泛化
#   3. 梯度惩罚独立化（ToT, NeurIPS 2021）：
#      从 pixel loss 中解耦梯度场 L1 约束，Stage 3 独立提高权重，
#      保护边缘不被新增损失模糊
# =========================================================

set_learning_rate(2e-4)

# ---------------------------------------------------------
# Stage 1: Epoch 0~39 — 基础结构学习
# 只用基础像素损失 + VGG 感知损失，LR 较高，充分探索参数空间
# ---------------------------------------------------------
opt.lambda_pixel = 1.0
opt.lambda_vgg = 0.1
opt.lambda_gan = 0.0
opt.lambda_maxrf = 0.0
opt.lambda_exclusion = 0.0
opt.lambda_laplacian = 0.0
opt.lambda_fea_decorr = 0.0
opt.lambda_vgg_ms = 0.0
opt.lambda_gradient = 0.0

print("[i] === V3 Training: 80 epochs, 3-stage curriculum ===")
print("[i] Stage 1 (Ep 0-39): Coarse Structural Separation")
print("[i]   lambda_pixel=1.0, lambda_vgg=0.1, LR=2e-4")

while engine.epoch < 80:
    # ---------------------------------------------------------
    # Stage 2: Epoch 40~59 — 精细结构解耦
    # 引入 MaxRF + 梯度互斥 + GAN + 轻量梯度惩罚
    # ---------------------------------------------------------
    if engine.epoch == 40:
        print("\n[i] === Stage 2 (Ep 40-59): Fine Structural Decoupling ===")
        set_learning_rate(1e-4)
        opt.lambda_maxrf = 0.3
        opt.lambda_exclusion = 0.05
        opt.lambda_gan = 0.01
        opt.lambda_gradient = 0.1    # 轻量独立梯度惩罚
        print("[i]   +MaxRF=0.3 +Exclusion=0.05 +GAN=0.01 +Gradient=0.1, LR=1e-4")

    # ---------------------------------------------------------
    # Stage 3: Epoch 60~79 — 真实场景精细化（反过拟合配置）
    # 核心改动：
    #   - 数据比例翻转为 [0.3合成, 0.7真实]，大量注入真实样本
    #   - Laplacian/FeaDecorr 权重减半，防止合成纹理套路固化
    #   - 新增多尺度感知损失，提供尺度不变的结构监督
    #   - 独立梯度惩罚翻倍，保护边缘细节
    # ---------------------------------------------------------
    if engine.epoch == 60:
        print("\n[i] === Stage 3 (Ep 60-79): Real-Data-Focused Polish ===")

        # 核心改动：翻转数据比例，70% 真实数据
        ratio = [0.3, 0.7]
        print("[i]   Flipping fusion ratio to {} (70% real data)".format(ratio))
        train_dataset_fusion.fusion_ratios = ratio

        set_learning_rate(5e-5)
        opt.lambda_maxrf = 0.5           # 加强 MaxRF
        opt.lambda_laplacian = 0.05      # 减半（v2=0.1），防止过拟合
        opt.lambda_fea_decorr = 0.005    # 减半（v2=0.01），防止过拟合
        opt.lambda_vgg_ms = 0.05         # 新增：多尺度感知损失
        opt.lambda_gradient = 0.2        # 翻倍，保护边缘
        print("[i]   +Laplacian=0.05 +FeaDecorr=0.005 +VGG_MS=0.05 +Gradient=0.2")
        print("[i]   lambda_maxrf=0.5, LR=5e-5")

    if engine.epoch == 70:
        set_learning_rate(2e-5)

    # 训练单步
    engine.train(train_dataloader_fusion)
    
    # 验证逻辑：每 5 epoch 评估一次
    if engine.epoch % 5 == 0:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20')