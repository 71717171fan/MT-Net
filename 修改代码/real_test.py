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


        test_set = TestDatasetFromFolder55('D:/AProject/meta/train_mata/real-images')

        test_dataloader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False,
                                     pin_memory=False)
        test_bar = tqdm(test_dataloader, desc='[testing datasets]')
else:
    print('\tInvalid input dataset name at CNN_train()')
    exit(1)

prepro=pre(0.5,0)

meta = Meta(8).to(device)
meta.load_state_dict(torch.load("D:/AProject/meta/train_mata/teacher_model/meta/meta_100.pth", map_location=device))
model = TransformNet(32)
model.load_state_dict(torch.load('D:/AProject/meta/train_mata/meta.pth', map_location=device))
model = model.to(device)

# model
gpuID = 0
torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

eps = 1e-10
start_time = time.time()
tt = 0

for i,(input)  in enumerate(test_dataloader):

        lr_patch = Variable(input, requires_grad=False).cuda(gpuID)

        unloader = transforms.ToPILImage()
        image = input.cpu().clone()
        image = image.squeeze(0)
        haze = unloader(image)
        x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

        pre_haze = prepro(x).cuda(gpuID)

        t0 = time.time()
        fea1, fea2, fea3, fea4, fea5, fea6 = meta(pre_haze)
        haze_features = cat((fea1, fea2, fea3, fea4, fea5), 1)

        lr_patch = lr_patch.to(device)
        haze_features = haze_features.to(device)
        pre_haze = pre_haze.to(device)
        output= model(lr_patch, haze_features, pre_haze)
        t1 = time.time()
        tt = tt + t1- t0
        # print(str((t1 - t0)))

        vutils.save_image(output.data,'D:/AProject/meta/train_mata/real_our/output/%05d.png' % (int(i)), padding=0,normalize=True)


