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


set_learning_rate(2e-4)


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

    if engine.epoch == 40:
        print("\n[i] === Stage 2 (Ep 40-59): Fine Structural Decoupling ===")
        set_learning_rate(1e-4)
        opt.lambda_maxrf = 0.3
        opt.lambda_exclusion = 0.05
        opt.lambda_gan = 0.01
        opt.lambda_gradient = 0.1   
        print("[i]   +MaxRF=0.3 +Exclusion=0.05 +GAN=0.01 +Gradient=0.1, LR=1e-4")


    if engine.epoch == 60:
        print("\n[i] === Stage 3 (Ep 60-79): Real-Data-Focused Polish ===")


        ratio = [0.3, 0.7]
        print("[i]   Flipping fusion ratio to {} (70% real data)".format(ratio))
        train_dataset_fusion.fusion_ratios = ratio

        set_learning_rate(5e-5)
        opt.lambda_maxrf = 0.5           
        opt.lambda_laplacian = 0.05      
        opt.lambda_fea_decorr = 0.005    
        opt.lambda_vgg_ms = 0.05         
        opt.lambda_gradient = 0.2        
        print("[i]   +Laplacian=0.05 +FeaDecorr=0.005 +VGG_MS=0.05 +Gradient=0.2")
        print("[i]   lambda_maxrf=0.5, LR=5e-5")

    if engine.epoch == 70:
        set_learning_rate(2e-5)

    engine.train(train_dataloader_fusion)
    
    if engine.epoch % 5 == 0:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20')