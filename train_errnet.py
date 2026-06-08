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
# 课程学习策略：所有 lambda 变量现在直接控制 backward_G 中的损失权重
# （已修复此前 lambda_l1 空转、双重更新、LR 反复升降的问题）
# =========================================================

set_learning_rate(1e-4)

# ---------------------------------------------------------
# Stage 1: Epoch 0~19 — 粗结构分离
# 只用基础像素损失 + VGG 感知损失，让网络先学会大结构
# ---------------------------------------------------------
opt.lambda_pixel = 1.0
opt.lambda_vgg = 0.1
opt.lambda_gan = 0.0
opt.lambda_maxrf = 0.0
opt.lambda_exclusion = 0.0
opt.lambda_laplacian = 0.0
opt.lambda_fea_decorr = 0.0

print("[i] Starting Stage 1: Coarse Structural Separation (Epochs 0-19)")

while engine.epoch < 60:
    # ---------------------------------------------------------
    # Stage 2: Epoch 20~39 — 精细结构解耦
    # 引入 MaxRF 加权损失 + 梯度互斥 + GAN，分离纹理边界
    # ---------------------------------------------------------
    if engine.epoch == 20:
        print("\n[i] Entering Stage 2: Fine Structural Decoupling (Epochs 20-39)")
        opt.lambda_pixel = 0.8          # 略微降低基础像素损失权重
        opt.lambda_maxrf = 0.5          # 引入 MaxRF 掩膜加权损失
        opt.lambda_exclusion = 0.05     # 激活梯度互斥损失
        opt.lambda_gan = 0.01           # 引入对抗损失

    if engine.epoch == 30:
        set_learning_rate(5e-5)

    # ---------------------------------------------------------
    # Stage 3: Epoch 40~59 — 高频细节打磨
    # 激活拉普拉斯边缘损失 + 特征去相关，锐化背景细节
    # ---------------------------------------------------------
    if engine.epoch == 40:
        print("\n[i] Entering Stage 3: High-Frequency Polish (Epochs 40-59)")
        set_learning_rate(1e-5)
        opt.lambda_pixel = 0.5          # 进一步降低像素损失权重，避免过度平滑
        opt.lambda_maxrf = 0.8          # 加强 MaxRF 关注强反光区
        opt.lambda_laplacian = 0.1      # 激活拉普拉斯边缘损失
        opt.lambda_fea_decorr = 0.01    # 引入特征去相关损失

    if engine.epoch == 45:
        ratio = [0.5, 0.5]
        print('[i] adjust fusion ratio to {}'.format(ratio))
        train_dataset_fusion.fusion_ratios = ratio

    if engine.epoch == 50:
        set_learning_rate(5e-6)

    # 训练单步
    engine.train(train_dataloader_fusion)
    
    # 验证逻辑
    if engine.epoch % 5 == 0:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20')