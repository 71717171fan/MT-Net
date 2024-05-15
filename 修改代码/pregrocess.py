import cv2 as cv

from PIL import Image
from PIL import ImageEnhance
# from utils import *
import torchvision.transforms as transforms
import numpy as np

from torch import nn, cat

#图像的预处理操作阶段



# enhance增强图像对比度，提升图像视觉效果
class enhance(nn.Module):

    def __init__(self,a,b):
        super(enhance, self).__init__()
        self.a = a
        self.b = b

    def forward(self, img):
        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.5
        img_ec = enh_con.enhance(contrast)


        #y = np.float(self.a) * img + self.b
        #y[y > 255] = 255
        #y = np.round(y)
        #img_ec = y.astype(np.uint8)

        img_ec = transforms.ToTensor()(img_ec)


        return img_ec

#平衡图像颜色
class balance(nn.Module):

    def __init__(self):
        super(balance, self).__init__()


    def forward(self, img):
        r, g, b = cv.split(img)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]
        # 求各个通道所占增益
        k = (r_avg + g_avg + b_avg) / 3
        #计算各个通道平均值的平均值作为整体增益。
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg
        #计算每个通道的增益
        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        #根据计算得到的绿色通道增益对绿色通道进行加权处理。
        balance_img = cv.merge([b, g, r])
        #将处理后的三个通道重新合并成一个图像
        balance_img = transforms.ToTensor()(balance_img)
        return balance_img

#Gamma校正处理
class gamma(nn.Module):

    def __init__(self):
        super(gamma, self).__init__()


    def forward(self, img):
        img_norm = img / 255.0  # 注意255.0得采用浮点数，对图像的像素值归一化处理
        img_gamma = np.power(img_norm, 2.5) * 255.0
        img_gamma = img_gamma.astype(np.uint8)
        img_gamma = transforms.ToTensor()(img_gamma)

        return img_gamma

#将上面的步骤合并，得到合并后的结果
class pre(nn.Module):

    def __init__(self,a,b):
        super(pre, self).__init__()
        self.a = a
        self.b = b

    def forward(self, img):

        #对输入图像进行线性变换
        np.float = float
        y = np.float(self.a) * img + self.b
        #图片像素值不大于255
        y[y > 255] = 255
        #像素值转为整数
        y = np.round(y)
        img_bright = y.astype(np.uint8)

        img_norm = img / 255.0  # 注意255.0得采用浮点数
        img_gamma = np.power(img_norm, 2.5) * 255.0
        img_gamma = img_gamma.astype(np.uint8)

        r, g, b = cv.split(img)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]
        # 求各个通道所占增益
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg
        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        balance_img = cv.merge([b, g, r])

        balance_img = transforms.ToTensor()(balance_img)
        img_bright = transforms.ToTensor()(img_bright)
        img_gamma = transforms.ToTensor()(img_gamma)
        img = transforms.ToTensor()(img)

        hazef = cat([img,balance_img, img_bright, img_gamma], 0)

        height = hazef.shape[1]
        width = hazef.shape[2]
        hazef = hazef.expand(1, 12, height, width)


        return  hazef