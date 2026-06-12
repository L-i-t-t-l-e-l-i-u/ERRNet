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

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
datadir = './datasets/processed_data'
raw_datadir = './datasets/raw_data'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')
datadir_unaligned = join(raw_datadir, 'Dataset/DSLR/unaligned_train250')

train_dataset = datasets.CEILDataset(datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size)
train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_unaligned = datasets.CEILTestDataset(datadir_unaligned, enable_transforms=True, flag={'unaligned':True}, size=None)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_unaligned, train_dataset_real], [0.25,0.5,0.25])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=True)

# 评估数据集
eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))
eval_dataset_real = datasets.CEILTestDataset(join(datadir, 'real20'), size=20, max_long_edge=512)
eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

engine = Engine(opt)
opt.save_epoch_freq = 1

def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        util.set_opt_param(optimizer, 'lr', lr)

# =========================================================
# 未对齐 Fine-tune（改进版）
#
# 关键改进：
#   1. 极低初始 LR（5e-6），保护已收敛的权重不被打散
#   2. 保留 v3 的损失函数权重——对齐数据（25%合成+25%真实）
#      仍走多损失约束，未对齐数据（50%）走 CX 损失
#   3. 每 5 epoch 评估 CEILNet + real20，监控是否过拟合
# =========================================================

# 沿用 v3 Stage 3 的损失函数配置（对齐数据上生效）
opt.lambda_pixel = 1.0
opt.lambda_vgg = 0.1
opt.lambda_gan = 0.01
opt.lambda_maxrf = 0.5
opt.lambda_exclusion = 0.05
opt.lambda_laplacian = 0.05
opt.lambda_fea_decorr = 0.005
opt.lambda_vgg_ms = 0.05
opt.lambda_gradient = 0.2

set_learning_rate(5e-6)

print("[i] === Unaligned Fine-tune (Improved) ===")
print("[i] LR=5e-6, data ratio: [0.25syn, 0.5unaligned, 0.25real]")
print("[i] Aligned samples: all v3 loss functions active")
print("[i] Unaligned samples: CX loss (unaligned_loss=vgg)")

while engine.epoch < 100:  # 80→100，做 20 epoch fine-tune
    if engine.epoch == 90:
        set_learning_rate(2e-6)

    engine.train(train_dataloader_fusion)
    
    if engine.epoch % 5 == 0:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20')