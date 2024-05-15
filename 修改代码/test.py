#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as ski_ssim
from model200314 import *
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
#from data_load_own import get_training_set, get_test_set
#from data_load_mix import get_dataset_deform
import argparse
from pregrocess import pre
from data_utils0210 import  TestDatasetFromFolder2,TestDatasetFromFolder,TestDatasetFromFolder4
from tqdm import tqdm
import time
from skimage import transform
from PIL import ImageFile
import torch
import warnings
# import visdom
from data_utils0210 import TestDatasetFromFolder55
from modelmeta import *

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Operation-wise Attention Network')
parser.add_argument('--gpu_num', '-g', type=int, default=0, help='Num. of GPUs')
parser.add_argument('--mode', '-m', default='yourdata', help='Mode (mix / yourdata)')

parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for training')
parser.add_argument('--max_epoch', type=int, default=10, help='Maximum number of epochs for training')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load dataset
if args.mode == 'mix' or args.mode == 'yourdata':
    if args.mode == 'mix':
        num_work = 8
        train_dir = '/dataset/train/'
        val_dir = '/dataset/val/'
        test_dir = './dataset/test/'
    elif args.mode == 'yourdata':
        num_work = 1
        # test_set = TestDatasetFromFolder4('/home/omnisky/4t/RESIDE/RTTS/RTTS')#

        test_set = TestDatasetFromFolder55('D:/AProject/meta/train_mata/real-images')
        # test_dataloader = DataLoader(dataset=test_set, num_workers=num_work, batch_size=1, shuffle=False, pin_memory=False)
        test_dataloader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False,
                                     pin_memory=False)
        test_bar = tqdm(test_dataloader, desc='[testing datasets]')
else:
    print('\tInvalid input dataset name at CNN_train()')
    exit(1)

prepro=pre(0.5,0)
# meta = torch.load("./dehaze_80.pth",map_location="cuda:0")#,map_location="cuda:0"
meta = Meta(8).to(device)
meta.load_state_dict(torch.load("D:/AProject/meta/train_mata/teacher_model/meta/meta_100.pth", map_location=device))
model = TransformNet(32)
model.load_state_dict(torch.load('D:/AProject/meta/train_mata/student_model_t1_100.pth', map_location=device))


# model
gpuID = 0
torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# if not os.path.exists('./results_r'):  #save path
#     os.makedirs('./results_r/Inputs')
#     os.makedirs('./results_r/Outputs')
#     os.makedirs('./results_r/Targets')


# test_ite = 0
# test_psnr = 0
# test_ssim = 0
eps = 1e-10
start_time = time.time()
tt = 0
# vis = visdom.Visdom()

# 创建用于显示 PSNR 和 SSIM 的窗口
# psnr_window = vis.line(X=[0], Y=[0], opts=dict(title='PSNR'))
# ssim_window = vis.line(X=[0], Y=[0], opts=dict(title='SSIM'))

# max_psnr = 0
# max_ssim = 0
# ************存储路径************
# output_folder = './results_r/outputs/'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# for i,(epoch) in range(args.start_epoch, args.max_epoch + 1):
#     current_max_psnr = 0
#     current_max_ssim = 0
for i,(input)  in enumerate(test_dataloader):
    # output = model(input)
    #     print('结果是'+str(input.size()))
        # image_name = image_name[0]
        lr_patch = Variable(input, requires_grad=False).cuda(gpuID)

        unloader = transforms.ToPILImage()
        image = input.cpu().clone()
        image = image.squeeze(0)
        haze = unloader(image)
        x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

        pre_haze = prepro(x).cuda(gpuID)
    # start_time = time.time()
        t0 = time.time()
        fea1, fea2, fea3, fea4, fea5, fea6 = meta(pre_haze)
        haze_features = cat((fea1, fea2, fea3, fea4, fea5), 1)
    # start_time = time.time()
        lr_patch = lr_patch.to(device)
        haze_features = haze_features.to(device)
        pre_haze = pre_haze.to(device)
        output,a,c,v,b,n = model(lr_patch, haze_features, pre_haze)
        t1 = time.time()
        tt = tt + t1- t0
        print(str((t1 - t0)))
    # save images
    # vutils.save_image(output.data, './results_r/Outputs/' + image_name, padding=0, normalize=True)
    # vutils.save_image(lr_patch.data, './results_r/Inputs/' + image_name, padding=0, normalize=True)
    # vutils.save_image(hr_patch.data, './results_r/Targets/'+ image_name, padding=0, normalize=True)
    #     file_path = os.path.join(output_folder, image_name)
        # vutils.save_image(output.data, file_path, padding=0, normalize=True)
        vutils.save_image(output.data,'D:/AProject/meta/train_mata/real_our/output/%05d.png' % (int(i)), padding=0,normalize=True)

#         output = output.data.cpu().numpy()[0]
#         output[output>1] = 1
#         output[output<0] = 0
#         output = output.transpose((1,2,0))
#     #print(output)
#         hr_patch = hr_patch.data.cpu().numpy()[0]
#         hr_patch[hr_patch>1] = 1
#         hr_patch[hr_patch<0] = 0
#         hr_patch = hr_patch.transpose((1,2,0))
#     # SSIM
#         ssim = ski_ssim(output, hr_patch, data_range=1, multichannel=True)
#         if ssim > current_max_ssim:
#           current_max_ssim = ssim
#         test_ssim+=ssim
#     # PSNR
#         imdf = (output - hr_patch) ** 2
#         mse = np.mean(imdf) + eps
#         psnr = 10 * math.log10(1.0/mse)
#         if psnr > current_max_psnr:
#            current_max_psnr = psnr
#         test_psnr+= psnr
#         test_ite += 1
#         print('PSNR: {:.4f}'.format(psnr))
#         print('SSIM: {:.4f}'.format(ssim))
#         torch.cuda.empty_cache()
#     print(f'Epoch {epoch} - Max PSNR: {current_max_psnr:.4f}, Max SSIM: {current_max_ssim:.4f}')
#
#     # 保存每一轮的最大 PSNR 和 SSIM
#     if current_max_psnr > max_psnr:
#         max_psnr = current_max_psnr
#     if current_max_ssim > max_ssim:
#         max_ssim = current_max_ssim
#     # vis.line(X=[epoch], Y=[current_max_psnr], win=psnr_window, update='append', name='Max PSNR')
#     # vis.line(X=[epoch], Y=[current_max_ssim], win=ssim_window, update='append', name='Max SSIM')
# # 输出整个训练过程中的最大 PSNR 和 SSIM
# print(f'Max PSNR over all epochs: {max_psnr:.4f}')
# print(f'Max SSIM over all epochs: {max_ssim:.4f}')










