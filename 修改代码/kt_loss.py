
from torchvision import models
import torch.nn.functional as F
from skimage import color
import torch
import numpy as np
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torchvision import transforms
import cv2



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#对比损失，分别传入结果、正向图像、负向图像
class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


#psnr.ssim
from typing import List, Optional, Tuple, Union
import torch
from utils import gaussian_filter


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1.0,
         reduction: Optional[str] = 'mean', convert_to_greyscale: bool = False):
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.
    Args:
        x: Batch of predicted images with shape (batch_size x channels x H x W)
        y: Batch of target images with shape  (batch_size x channels x H x W)
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        convert_to_greyscale: Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.

    Returns:
        PSNR: Index of similarity betwen two images.
    Note:
        Implementaition is based on Wikepedia https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Reference:
        https://github.com/photosynthesis-team/piq/blob/master/piq/psnr.py
    """

    # Constant for numerical stability
    EPS = 1e-8

    x = x / data_range
    y = y / data_range

    if (x.size(1) == 3) and convert_to_greyscale:
        # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(x)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score = - 10 * torch.log10(mse + EPS)

    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
         k1: float = 0.01, k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Interface of Structural Similarity (SSIM) index.
    Args:
        x: Batch of images. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        y: Batch of images. Required to be 2D (H, W), 3D (C,H,W) 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        full: Return cs map or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.
    References:
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`

           https://github.com/photosynthesis-team/piq/blob/master/piq/ssim.py
    """

    if isinstance(x, torch.ByteTensor) or isinstance(y, torch.ByteTensor):
        x = x.type(torch.float32)
        y = y.type(torch.float32)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    if reduction != 'none':
        reduction_operation = {'mean': torch.mean,
                               'sum': torch.sum}
        ssim_val = reduction_operation[reduction](ssim_val, dim=0)
        cs = reduction_operation[reduction](cs, dim=0)

    if full:
        return ssim_val, cs

    return ssim_val

def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.
    Args:
        x: Batch of images, (N,C,H,W).
        y: Batch of images, (N,C,H,W).
        kernel: 2D Gaussian kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """

    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    n_channels = x.size(1)
    mu1 = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2 = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    compensation = 1.0
    sigma1_sq = compensation * (F.conv2d(x * x, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq)
    sigma2_sq = compensation * (F.conv2d(y * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq)
    sigma12 = compensation * (F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2)

    # Set alpha = beta = gamma = 1.
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    ssim_val = ssim_map.mean(dim=(-1, -2))
    cs = cs_map.mean(dim=(-1, -2))
    return ssim_val, cs

def _ssim_per_channel_complex(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                              data_range: Union[float, int] = 1., k1: float = 0.01,
                              k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.
    Args:
        x: Batch of complex images, (N,C,H,W,2).
        y: Batch of complex images, (N,C,H,W,2).
        kernel: 2-D gauss kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    sigma12_real = F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_real
    sigma12_imag = F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    # Set alpha = beta = gamma = 1.
    cs_map = (sigma12 * 2 + c2 * compensation) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation)
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation)
    ssim_map = ssim_map * cs_map

    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs




def CAPLoss(img):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    unloader = transforms.ToPILImage()
    image = img.cpu().clone()
    image = image.squeeze(0)
    haze = unloader(image)
    x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

    HSV_img = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    image = np.asarray(HSV_img)
    H, S, V = cv2.split(image)
    # print(img)
    # print(S - V)
    totensor = transforms.ToTensor()
    S = totensor(S)
    V = totensor(V)
    l1loss = L1Loss()
    loss = l1loss(S, V)
    # print(loss)



    # loss = L1Loss(size_average=True)(S, V)
    return loss


    # loss.backward()




def LabLoss(dehaze,hazy):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    unloader = transforms.ToPILImage()
    image = dehaze.cpu().clone()
    image = image.squeeze(0)
    dehaze = unloader(image)

    unloader = transforms.ToPILImage()
    image = hazy.cpu().clone()
    image = image.squeeze(0)
    hazy = unloader(image)
    # x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)
    #
    # HSV_img = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    # image = np.asarray(HSV_img)
    # H, S, V = cv2.split(image)
    # # print(img)
    # # print(S - V)
    totensor = transforms.ToTensor()
    # S = totensor(S)
    # V = totensor(V)
    # l1loss = L1Loss()
    # loss = l1loss(S, V)
    # print(loss)
    # skimage.color
    lab_dehaze=color.rgb2lab(dehaze)
    ldehaze, a, b = cv2.split(lab_dehaze)
    lab_hazy = color.rgb2lab(hazy)
    lhazy, a, b = cv2.split(lab_hazy)
    ldehaze = totensor(ldehaze)
    lhazy = totensor(lhazy )
    l1loss = L1Loss()
    loss = l1loss(ldehaze, lhazy )



    # loss = L1Loss(size_average=True)(S, V)
    return loss


#色彩损失
import numpy as np
from PIL import Image

def calculate_average_pixel_difference(image1, image2):
    # 读取图片

    # 将张量转换为 NumPy 数组
    image11 = image1.detach().cpu().numpy()[0]
    # 将张量转换为 NumPy 数组
    image22 = image2.detach().cpu().numpy()[0]

    # 将图片转换为numpy数组
    array1 = np.array(image11)
    array2 = np.array(image22)

    # 确保两张图片具有相同的尺寸
    if array1.shape != array2.shape:
        raise ValueError("两张图片的尺寸不一致")

    # 计算每个像素点之间的色彩差的绝对值
    pixel_diff = np.abs(array1 - array2)

    # 将RGB通道的差值相加并求平均
    average_diff = np.mean(pixel_diff)
    result = average_diff/100

    return result


import torch
import torch.nn.functional as F

def contrastive_loss(image1, image2):
    # 计算图像的均值
    mean1 = torch.mean(image1)
    mean2 = torch.mean(image2)

    # 计算图像的方差
    var1 = torch.var(image1)
    var2 = torch.var(image2)

    # 计算对比度损失
    loss = torch.abs(var1 - var2)

    return loss


class C2R(nn.Module):
    def __init__(self, ablation=False):

        super(C2R, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n1, n2, n1_weight, n2_weight):
        a_vgg, p_vgg, n1_vgg, n2_vgg = self.vgg(a), self.vgg(p), self.vgg(n1), self.vgg(n2)

        loss = 0

        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an1 = self.l1(a_vgg[i], n1_vgg[i].detach())
                d_an2 = self.l1(a_vgg[i], n2_vgg[i].detach())

                contrastive = d_ap / (d_an1 * n1_weight + d_an2 * n2_weight + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive

        return loss