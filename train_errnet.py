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
# 课程学习策略 v2（修复训练量不足问题）
#
# 问题诊断：
#   1. 移除 monkey-patch 双重更新 → 每 epoch 有效训练量减半
#   2. batch_size=4 → 每 epoch 更新次数只有 batch_size=1 的 25%
#   3. LR 衰减过快 → 模型还没收敛就被"刹车"
#
# 修复方案：
#   - 初始 LR 提高到 2e-4，补偿更新次数不足
#   - lambda_pixel 全程保持 1.0（匹配原始代码行为）
#   - 总训练延长到 80 epoch
#   - 课程阶段推迟，给基础训练更多时间
# =========================================================

set_learning_rate(2e-4)

# ---------------------------------------------------------
# Stage 1: Epoch 0~39 — 基础结构学习
# 只用基础像素损失 + VGG 感知损失，让网络先学好大结构
# lambda_pixel=1.0 匹配原始代码行为，不做任何衰减
# ---------------------------------------------------------
opt.lambda_pixel = 1.0
opt.lambda_vgg = 0.1
opt.lambda_gan = 0.0
opt.lambda_maxrf = 0.0
opt.lambda_exclusion = 0.0
opt.lambda_laplacian = 0.0
opt.lambda_fea_decorr = 0.0

print("[i] Starting Stage 1: Coarse Structural Separation (Epochs 0-39)")
print("[i] lambda_pixel=1.0, lambda_vgg=0.1, LR=2e-4")

while engine.epoch < 80:
    # ---------------------------------------------------------
    # Stage 2: Epoch 40~59 — 精细结构解耦
    # 引入 MaxRF 加权损失 + 梯度互斥 + GAN
    # ---------------------------------------------------------
    if engine.epoch == 40:
        print("\n[i] Entering Stage 2: Fine Structural Decoupling (Epochs 40-59)")
        set_learning_rate(1e-4)
        opt.lambda_maxrf = 0.3          # 引入 MaxRF 掩膜加权损失
        opt.lambda_exclusion = 0.05     # 激活梯度互斥损失
        opt.lambda_gan = 0.01           # 引入对抗损失

    # ---------------------------------------------------------
    # Stage 3: Epoch 60~79 — 高频细节打磨
    # 激活拉普拉斯边缘损失 + 特征去相关，锐化背景细节
    # ---------------------------------------------------------
    if engine.epoch == 60:
        print("\n[i] Entering Stage 3: High-Frequency Polish (Epochs 60-79)")
        set_learning_rate(5e-5)
        opt.lambda_maxrf = 0.5          # 加强 MaxRF 关注强反光区
        opt.lambda_laplacian = 0.1      # 激活拉普拉斯边缘损失
        opt.lambda_fea_decorr = 0.01    # 引入特征去相关损失

    if engine.epoch == 70:
        set_learning_rate(2e-5)

    # 训练单步
    engine.train(train_dataloader_fusion)
    
    # 验证逻辑：每 5 epoch 评估一次
    if engine.epoch % 5 == 0:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20')