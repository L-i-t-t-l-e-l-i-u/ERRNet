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

# 初始学习率设置
set_learning_rate(1e-4)

# 【核心修改区：Curriculum Learning Strategy 初始化】
# 假设你在 engine.model.opt 中定义了这些权重变量。如果没有，请在 engine.py 中添加对它们的解析和损失计算。
engine.model.opt.lambda_gan = 0.0          # GAN Loss 权重
engine.model.opt.lambda_l1 = 1.0           # 基础/MaxRF L1 Loss 权重
engine.model.opt.lambda_exclusion = 0.0    # 梯度互斥 Loss 权重
engine.model.opt.lambda_laplacian = 0.0    # 拉普拉斯边缘 Loss 权重

print("[i] Starting Stage 1: Coarse Structural Separation (Epochs 1-20)")

while engine.epoch < 60:
    # -------------------------------------------------------------
    # Stage 2: Epoch 21~40 (Fine Structural Decoupling)
    # -------------------------------------------------------------
    if engine.epoch == 20:
        print("\n[i] Entering Stage 2: Fine Structural Decoupling")
        engine.model.opt.lambda_gan = 0.01        # 引入对抗损失提升真实感
        engine.model.opt.lambda_exclusion = 0.05  # 激活 Pseudo-Reflection Gradient Exclusion，分离边界
        # 保持 L1 为主导
        
    if engine.epoch == 30:
        set_learning_rate(5e-5)

    # -------------------------------------------------------------
    # Stage 3: Epoch 41~60 (High-Frequency Polish)
    # -------------------------------------------------------------
    if engine.epoch == 40:
        print("\n[i] Entering Stage 3: High-Frequency Polish")
        set_learning_rate(1e-5)
        engine.model.opt.lambda_l1 = 0.5          # 降低基础像素 L1 的比重，避免平滑
        engine.model.opt.lambda_laplacian = 0.1   # 满血激活 Laplacian Edge Loss，锐化背景细节
        
    if engine.epoch == 45:
        ratio = [0.5, 0.5]
        print('[i] adjust fusion ratio to {}'.format(ratio))
        train_dataset_fusion.fusion_ratios = ratio
        set_learning_rate(5e-5) # 稍微回弹LR以适应数据分布变化
        
    if engine.epoch == 50:
        set_learning_rate(1e-5)

    # 训练单步
    engine.train(train_dataloader_fusion)
    
    # 验证逻辑
    if engine.epoch % 5 == 0:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20')