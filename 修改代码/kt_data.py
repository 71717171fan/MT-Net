from os import listdir
from os.path import join
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage, CenterCrop, Resize, ToTensor
import natsort
import torch.backends.cudnn as cudnn
from PIL import ImageFile
import warnings
import torch
import torch.utils.data as data
import cv2
import torchvision.transforms as transforms
from PIL import Image



cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import transforms.pix2pix as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp', '.BMP'])


def calculate_valid_crop_size(crop_size):
    return crop_size


def train_h_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),

        ToTensor()

    ])


def train_s_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),

    ])


def test_h_transform():
    return transforms.Compose([

        ToTensor(),

    ])


def test_s_transform():
    return transforms.Compose([

        ToTensor(),

    ])


def display_transform():
    return Compose([
        ToPILImage(),
        # crop((0,0,1000,600)),
        # CenterCrop(500),
        ToTensor()
    ])

#用于教师网络
class TrainDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir_h, dataset_dir_s, crop_size):
        super(TrainDatasetFromFolder1, self).__init__()
        # self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:350] if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:10] for p in range(35) if is_image_file(x)]

        self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h)) if
                                  is_image_file(x)]

        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s)) for p in
        #                           range(10) if is_image_file(x)]
        self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s)) if
                                  is_image_file(x)]

        crop_size = calculate_valid_crop_size(crop_size)
        self.h_transform = train_h_transform(crop_size)
        self.s_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        h_image = self.h_transform(Image.open(self.image_filenames_h[index]))
        s_image = self.s_transform(Image.open(self.image_filenames_s[index]))

        return h_image, s_image

    def __len__(self):
        return len(self.image_filenames_h)
#用于教师网络
class TestDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder1, self).__init__()
        # self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/clear'  # '/Outputs'#'/hazy'#'/Outputs'
        self.s_path = dataset_dir + '/clear'  # '/clear'#'/Targets'#'/gt'#'/Targets'
        # self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if
                            is_image_file(x)]  # sorted
        # self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) for p in range(10) if
        #                     is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if
                            is_image_file(x)]
        # self.h_transform = test_h_transform()
        # self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return ToTensor()(h_image), ToTensor()(s_image)  # ToTensor()(h_image)

    def __len__(self):
        return len(self.h_filenames)
class TestDatasetFromFolder8(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder8, self).__init__()
        # self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/haze'  # '/Outputs'#'/hazy'#'/Outputs'
        self.s_path = dataset_dir + '/clear'  # '/clear'#'/Targets'#'/gt'#'/Targets'
        # self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if
                            is_image_file(x)]  # sorted
        # self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) for p in range(10) if
        #                     is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if
                            is_image_file(x)]
        # self.h_transform = test_h_transform()
        # self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return ToTensor()(h_image), ToTensor()(s_image)  # ToTensor()(h_image)

    def __len__(self):
        return len(self.h_filenames)
class TestDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder2, self).__init__()
        # self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/haze'  # '/Outputs'#'/hazy'#'/Outputs'
        self.s_path = dataset_dir + '/clear'  # '/clear'#'/Targets'#'/gt'#'/Targets'
        # self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if
                            is_image_file(x)]  # sorted
        # self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) for p in range(10) if
        #                     is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if
                            is_image_file(x)]
        # self.h_transform = test_h_transform()
        # self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return ToTensor()(h_image), ToTensor()(s_image)  # ToTensor()(h_image)

    def __len__(self):
        return len(self.h_filenames)
#用于教师网络
class TestDatasetFromFolder4(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder4, self).__init__()
        # self.h_path = dataset_dir + '/JPEGImages_new'  # S0501  JPEGImages_new
        self.h_path = dataset_dir + '/clear'  # S0501  JPEGImages_new
        # self.s_path = dataset_dir + '/JPEGImages_new'  # JPEGImages_new
        self.s_path = dataset_dir + '/clear'  # JPEGImages_new
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if
                            is_image_file(x)]  # sorted
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if
                            # for p in range(10)
                            is_image_file(x)]
        # crop_size = calculate_valid_crop_size(crop_size)
        # self.h_transform = test_h_transform(crop_size)
        # self.s_transform = test_s_transform(crop_size)

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        # h_image = self.h_transform(Image.open(self.h_filenames[index]))
        # s_image = self.s_transform(Image.open(self.s_filenames[index]))
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])

        return image_name, ToTensor()(h_image), ToTensor()(s_image)

    def __len__(self):
        return len(self.h_filenames)


#用于学生网络
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        # self.h_path = dataset_dir + '/JPEGImages_new'  # S0501  JPEGImages_new
        self.h_path = dataset_dir + '/haze'  # S0501  JPEGImages_new
        # self.s_path = dataset_dir + '/JPEGImages_new'  # JPEGImages_new
        self.s_path = dataset_dir + '/clear'  # JPEGImages_new
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if
                            is_image_file(x)]  # sorted
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if
                            # for p in range(10)
                            is_image_file(x)]
        # crop_size = calculate_valid_crop_size(crop_size)
        # self.h_transform = test_h_transform(crop_size)
        # self.s_transform = test_s_transform(crop_size)

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        # h_image = self.h_transform(Image.open(self.h_filenames[index]))
        # s_image = self.s_transform(Image.open(self.s_filenames[index]))
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])

        return ToTensor()(h_image), ToTensor()(s_image)

    def __len__(self):
        return len(self.h_filenames)


#y用于学生网络分类为训练集，测试集以及验证集
class DataGeneratorPaired(data.Dataset):

    def __init__(self, splits, mode="train"):

        if mode == "train":
            self.gt_paths = splits["tr_gt_paths"]
            self.hazy_paths = splits["tr_hazy_paths"]
        elif mode == "val":
            self.gt_paths = splits["val_gt_paths"]
            self.hazy_paths = splits["val_hazy_paths"]
        elif mode == "test":
            self.gt_paths = splits["test_gt_paths"]
            self.hazy_paths = splits["test_hazy_paths"]
        else:
            raise "Incorrect dataset mode"

        self.transform_hazy = self.get_transforms()
        self.transform_gt = self.get_transforms()

    def get_transforms(self):
        """
        Returns transforms to apply on images.
        """

        transforms_list = []

        transforms_list.append(transforms.CenterCrop(255))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return transforms.Compose(transforms_list)

    def __getitem__(self, index):

        gt = Image.open(self.gt_paths[index])
        hazy = Image.open(self.hazy_paths[index])

        if self.transform_hazy is not None:
            hazy = self.transform_hazy(hazy)

        if self.transform_gt is not None:
            gt = self.transform_gt(gt)

        item = {"hazy": hazy,
                "gt": gt,
                "gt_paths": self.gt_paths[index],
                "hazy_paths": self.hazy_paths[index]}

        return item

    def __len__(self):

        return len(self.gt_paths)

# class TestDatasetFromFolder4(Dataset):
#     def __init__(self, dataset_dir):
#         super(TestDatasetFromFolder4, self).__init__()
#         # self.h_path = dataset_dir + '/JPEGImages_new'  # S0501  JPEGImages_new
#         self.h_path = dataset_dir + '/haze'  # S0501  JPEGImages_new
#         # self.s_path = dataset_dir + '/JPEGImages_new'  # JPEGImages_new
#         self.s_path = dataset_dir + '/clear'  # JPEGImages_new
#         self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if
#                             is_image_file(x)]  # sorted
#         self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if
#                             # for p in range(10)
#                             is_image_file(x)]
#         # crop_size = calculate_valid_crop_size(crop_size)
#         # self.h_transform = test_h_transform(crop_size)
#         # self.s_transform = test_s_transform(crop_size)
#
#     def __getitem__(self, index):
#         hazy_paths = self.h_filenames[index].split('/')[-1]
#         gt_paths = self.s_filenames[index].split('/')[-1]
#         # h_image = self.h_transform(Image.open(self.h_filenames[index]))
#         # s_image = self.s_transform(Image.open(self.s_filenames[index]))
#         h_image = Image.open(self.h_filenames[index])
#         s_image = Image.open(self.s_filenames[index])
#
#         return hazy_paths, gt_paths,ToTensor()(h_image), ToTensor()(s_image)
#
#     def __len__(self):
#         return len(self.h_filenames)

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class DataGeneratorPaired(data.Dataset):

    def __init__(self, splits, mode="train"):

        if mode == "train":
            self.gt_paths = splits["tr_gt_paths"]
            self.hazy_paths = splits["tr_hazy_paths"]
        elif mode == "val":
            self.gt_paths = splits["val_gt_paths"]
            self.hazy_paths = splits["val_hazy_paths"]
        elif mode == "test":
            self.gt_paths = splits["test_gt_paths"]
            self.hazy_paths = splits["test_hazy_paths"]
        else:
            raise "Incorrect dataset mode"

        self.transform_hazy = self.get_transforms()
        self.transform_gt = self.get_transforms()

    def get_transforms(self):
        """
        Returns transforms to apply on images.
        """

        transforms_list = []

        transforms_list.append(transforms.CenterCrop(255))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return transforms.Compose(transforms_list)

    def __getitem__(self, index):

        gt = Image.open(self.gt_paths[index])
        hazy = Image.open(self.hazy_paths[index])



        if self.transform_hazy is not None:
            hazy = self.transform_hazy(hazy)

        if self.transform_gt is not None:
            gt = self.transform_gt(gt)

        item = {"hazy": hazy,
                "gt": gt,
                "gt_paths": self.gt_paths[index],
                "hazy_paths": self.hazy_paths[index]}

        return item

    def __len__(self):

        return len(self.gt_paths)

#用于加载学生网络第一轮结果
class TrainDatasetFromFolder8(Dataset):
    def __init__(self, dataset_dir_h,  crop_size):
        super(TrainDatasetFromFolder8, self).__init__()
        # self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:350] if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:10] for p in range(35) if is_image_file(x)]

        self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h)) if
                                  is_image_file(x)]

        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s)) for p in
        #                           range(10) if is_image_file(x)]

        crop_size = calculate_valid_crop_size(crop_size)
        self.h_transform = train_h_transform(crop_size)


    def __getitem__(self, index):
        h_image = self.h_transform(Image.open(self.image_filenames_h[index]))


        return h_image

    def __len__(self):
        return len(self.image_filenames_h)
