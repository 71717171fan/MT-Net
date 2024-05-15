import torch.nn.functional as F
import torchvision.transforms as transforms
from math import log10
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
from torch.autograd import Variable
from pregrocess import pre
from torch import cat
import numpy as np
import os
import glob
import configparser as cp
import socket
import math
import torch
import shutil
from torchvision.utils import save_image
import argparse


cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)
epsilon = 1e-5
prepro=pre(0.5,0)
#meta = torch.load("./model/dehaze_80.pth",map_location="cuda:0")#,map_location="cuda:0"  /home/omnisky/volume/3meta-opera
# meta = torch.load("./dehaze_80.pth",map_location="cuda:0")#,map_location="cuda:0"



#这些函数共同构成了一个完整的图像处理工具集
# 用于将图像转换为张量
# 在张量和图像之间进行转换
# 显示图像以及计算特征图的统计信息
def preprocess_image(image, target_width=None):
    """输入 PIL.Image 对象，输出标准化后的四维 tensor"""
    #对图像宽度进行操作，将照片变为所需宽度
    if target_width:
        t = transforms.Compose([
            transforms.Resize(target_width),
            transforms.CenterCrop(target_width),
            transforms.ToTensor(),
            tensor_normalizer,
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            tensor_normalizer,
        ])
    return t(image).unsqueeze(0)


def image_to_tensor(image, target_width=None):
    """输入 OpenCV 图像，范围 0~255，BGR 顺序，输出标准化后的四维 tensor"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return preprocess_image(image, target_width)


def read_image(path, target_width=None):
    """输入图像路径，输出标准化后的四维 tensor"""
    image = Image.open(path)
    return preprocess_image(image, target_width)


def recover_image(tensor):
    """输入 GPU 上的四维 tensor，输出 0~255 范围的三维 numpy 矩阵，RGB 顺序"""
    image = tensor.detach().cpu().numpy()
    image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + \
            np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]


def recover_tensor(tensor):
    m = torch.tensor(cnn_normalization_mean).view(1, 3, 1, 1).to(tensor.device)
    s = torch.tensor(cnn_normalization_std).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * s + m
    return tensor.clamp(0, 1)


def imshow(tensor, title=None):
    """输入 GPU 上的四维 tensor，然后绘制该图像"""
    image = recover_image(tensor)
    print(image.shape)
    plt.imshow(image)
    if title is not None:
        plt.title(title)


def mean_std(features):
    """输入 VGG16 计算的四个特征，输出每张特征图的均值和标准差，长度为1920"""
    mean_std_features = []
    for x in features:
        x = x.view(*x.shape[:2], -1)
        x = torch.cat([x.mean(-1), torch.sqrt(x.var(-1) + epsilon)], dim=-1)
        n = x.shape[0]
        x2 = x.view(n, 2, -1).transpose(2, 1).contiguous().view(n, -1)  # 【mean, ..., std, ...] to [mean, std, ...]
        mean_std_features.append(x2)
    mean_std_features = torch.cat(mean_std_features, dim=-1)
    return mean_std_features


class Smooth:
    # 对输入的数据进行滑动平均
    def __init__(self, windowsize=100):
        self.window_size = windowsize
        self.data = np.zeros((self.window_size, 1), dtype=np.float32)
        self.index = 0

    def __iadd__(self, x):
        if self.index == 0:
            self.data[:] = x
        self.data[self.index % self.window_size] = x
        self.index += 1
        return self

    def __float__(self):
        return float(self.data.mean())

    def __format__(self, f):
        return self.__float__().__format__(f)




def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6

#计算信噪比
def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
#    measure.compare_psnr()

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

#skimage库计算图像的结构相似度指标
def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list

#对验证数据集进行验证，计算平均PSNR和SSIM
def validation(metanet,dehaze_net, val_dataloader, device):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_dataloader):

        with torch.no_grad():
            h, s = val_data
            varIn = Variable(h)
            varTar = Variable(s)

            varIn = varIn.to(device)
            varTar = varTar.to(device)
            #batch_size = h.size(0)
            #running_results['batch_sizes'] += batch_size

            # 检查纯色
            x = varIn.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue

            unloader = transforms.ToPILImage()
            image = varIn.cpu().clone()
            image = image.squeeze(0)
            haze = unloader(image)
            x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)
            pre_haze = prepro(x).to(device)

            outs = []

            fea1, fea2, fea3, fea4, fea5, fea6 = meta(pre_haze)
            outs.append(fea1)
            outs.append(fea2)
            outs.append(fea3)
            outs.append(fea4)
            outs.append(fea5)

            haze_features = cat((fea1, fea2, fea3, fea4, fea5), 1)
            # print(haze_features.size())

            haze_mean_std = mean_std(outs)
            weights = metanet(haze_mean_std)
            dehaze_net.set_weights(weights, 0)

            # pre_haze = prepro(x).to(device)
            dehaze = dehaze_net(varIn, haze_features)


            #haze = haze.to(device)
            #gt = gt.to(device)
            #dehaze = net(haze)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, varTar ))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(dehaze, varTar ))

        # --- Save image --- #
        #if save_tag:
            #save_image(dehaze, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


#根据训练的epoch调整学习率
def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.8):

    # --- Decay learning rate --- #
    step = 5 if category == 'indoor' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))









def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def read_config():
    config = cp.ConfigParser()
    cur_path = os.path.abspath('C:/Users/86198/PycharmProjects/pythonProject/myself/meta/train_mata/config.ini')
    config.read(os.path.join(cur_path, 'config.ini'))
    # host = socket.gethostname()
    # return config[host]

    host = 'LAPTOP-L3FDGC24'
    path_dataset = config[host]['path_dataset']
    path_aux = config[host]['path_aux']

    return path_dataset, path_aux

def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x


def save_checkpoint(state, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)


def load_files_and_partition(root_path, train_ratio=0.9, val_ratio=0.05, hazy_dir_name="haze", gt_dir_name="clear"):
    np.random.seed(0)

    gt_paths, hazy_paths = load_pairs(root_path, hazy_dir_name=hazy_dir_name, gt_dir_name=gt_dir_name)


    hazy_paths = pair_selection(gt_paths, hazy_paths)

    if len(gt_paths) != len(hazy_paths):
        raise "Inconsistent dataset length"

    indexes = list(range(len(gt_paths)))

    tr_indexes = np.random.choice(indexes, int(train_ratio * len(indexes)), replace=False).astype(int)
    val_indexes = np.random.choice(np.setdiff1d(indexes, tr_indexes), int(val_ratio * len(indexes)),
                                   replace=False).astype(int)
    te_indexes = np.setdiff1d(indexes, np.union1d(tr_indexes, val_indexes)).astype(int)

    splits = dict()

    splits["tr_gt_paths"] = gt_paths[tr_indexes]
    splits["val_gt_paths"] = gt_paths[val_indexes]
    splits["test_gt_paths"] = gt_paths[te_indexes]

    splits["tr_hazy_paths"] = hazy_paths[tr_indexes]
    splits["val_hazy_paths"] = hazy_paths[val_indexes]
    splits["test_hazy_paths"] = hazy_paths[te_indexes]

    return splits


def get_last_part_of_path(paths):
    parted_paths = []

    for i in range(len(paths)):
        last_part = os.path.basename(os.path.normpath(paths[i]))

        parted_paths.append(last_part)

    return parted_paths


def pair_selection(gt_paths, hazy_paths):
    m = len(gt_paths)

    last_part_hazy_paths = get_last_part_of_path(hazy_paths)

    selected_hazy_paths = []

    for i in range(m):
        # get last part of the path
        name = os.path.basename(os.path.normpath(gt_paths[i])).split(".")[0].split("_")[0]

        # find matching names
        indexes = np.flatnonzero(np.core.defchararray.find(last_part_hazy_paths, name) != -1)

        # Select 1 pair from N matching examples
        selected_path = np.random.choice(hazy_paths[indexes], 1, replace=False)

        selected_hazy_paths.append(selected_path[0])

    return np.array(selected_hazy_paths)


def load_pairs(root_path, hazy_dir_name="hazy", gt_dir_name="GT"):
    return (load_gt_images(root_path, gt_dir_name), load_hazy_images(root_path, hazy_dir_name))


def load_gt_images(root_path, gt_dir_name):
    return np.array([name for name in glob.glob(os.path.join(root_path, gt_dir_name, "*.jpg"))])


def load_hazy_images(root_path, hazy_dir_name):
    return np.array([name for name in glob.glob(os.path.join(root_path, hazy_dir_name, "*.jpg"))])


def save_an_image(path, path_results, img, postfix="_REC"):
    *base, ext = os.path.basename(path).split(".")
    base = ".".join(base)

    base = base + postfix

    ext = "png"

    img_name = base + "." + ext
    path = os.path.join(path_results, img_name)

    save_image(img, path, normalize=True, range=(-1, 1))


def gaussian_filter(size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the lernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: 2D kernel with shape (1 x kernel_size x kernel_size)
    References:
        https://github.com/photosynthesis-team/piq/blob/master/piq/functional/filters.py
    """
    coords = torch.arange(size).to(dtype=torch.float32)
    coords -= (size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


if __name__ == '__main__':
    splits = load_files_and_partition('C:/Users/86198/PycharmProjects/pythonProject/myself/meta/train_mata/indoor/train')
    print("训练集模糊图像文件路径列表：")
    for path in splits["tr_hazy_paths"]:
        print(path)

    print("验证集模糊图像文件路径列表：")
    for path in splits["val_hazy_paths"]:
        print(path)

    print("测试集模糊图像文件路径列表：")
    for path in splits["test_hazy_paths"]:
        print(path)
