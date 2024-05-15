import argparse
import time
import math
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as ski_ssim
import csv
import logging
import utils
from kt_data import *
from tqdm import tqdm
from student_model import *
from torchvision.models import vgg16
from perceptual import *
from kt_loss import *
from utils import *
from modelmeta import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# meta = torch.load("./dehaze_80.pth", map_location=device)

meta = Meta(8).to(device)
meta.load_state_dict(torch.load('D:/AProject/meta/train_mata/teacher_model/meta/meta_100.pth',map_location = device))

prepro = pre(0.5, 0)
teacher = TransformNet(32).to(device)
teacher.load_state_dict(torch.load('D:/AProject/meta/train_mata/teacher_model/teacher_model1_100.pth',map_location = device))
# student = TransformNet(32).to(device)
# student.load_state_dict(torch.load('D:/AProject/meta/train_mata/student_try_model/xw.pth',map_location = device))
# # teacher.load_state_dict(torch.load('D:/ws/meta/train_mata/teacher_model/teacher_model_144.pth',map_location = device))


class CNN_train():
    def __init__(self, dataset_name, imgSize=256, batchsize=8):
        self.imgSize = imgSize
        self.batchsize = batchsize
        self.dataset_name = dataset_name

        # load dataset
        if dataset_name == 'mix' or dataset_name == 'yourdata':
            if dataset_name == 'mix':
                self.num_work = 8


            elif dataset_name == 'yourdata':
                self.num_work = 8
                #训练室内图像
                # path_dataset = 'C:/Users/86198/PycharmProjects/pythonProject/myself/meta/train_mata/outdoor/train'
                # splits = load_files_and_partition(path_dataset, train_ratio=0.8, val_ratio=0.05, hazy_dir_name="haze",
                #                                   gt_dir_name="clear")
                train_set = TrainDatasetFromFolder1('D:/AProject/meta/train_mata/indoor/train/haze',
                                                    'D:/AProject/meta/train_mata/indoor/train/clear', crop_size=self.imgSize)
                test_set = TestDatasetFromFolder2(
                    'D:/AProject/meta/train_mata/indoor/test')
                self.dataloader = DataLoader(dataset=train_set, num_workers=self.num_work, batch_size=self.batchsize,
                                             shuffle=True, drop_last=True)
                self.val_loader = DataLoader(dataset=test_set, num_workers=self.num_work, batch_size=self.batchsize,
                                             shuffle=True, drop_last=True)
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, epoch_num=100, gpu_num=1):
        print('GPUID    :', gpuID)
        print('epoch_num:', epoch_num)

        # define model
        torch.manual_seed(2018)
        torch.cuda.manual_seed(2018)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        L1_loss = nn.L1Loss()
        L1_loss = L1_loss.cuda(gpuID)
        # --- Define the perceptual loss network --- #
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False

        loss_network = LossNetwork(vgg_model).cuda(gpuID)
        loss_network.eval()
        teacher_model = TransformNet(32).to(device)

        if gpu_num > 1:
            device_ids = [i for i in range(gpu_num)]
            teacher_model = torch.nn.DataParallel(teacher_model, device_ids=device_ids)
        teacher_model = teacher_model.cuda(gpuID)
        logging.info("param size = %fMB", utils.count_parameters_in_MB(teacher_model))
        print('Param:', utils.count_parameters_in_MB(teacher_model))
        optimizer = optim.Adam(teacher_model.parameters(), lr=0.001, betas=(0.9, 0.999))
        meta_optimizer = optim.SGD(meta.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
        test_interval = 1  # 5

        for epoch in range(1, epoch_num + 1):
            scheduler.step()
            start_time = time.time()
            print('epoch', epoch)
            train_loss = 0
            #显示进度条
            train_bar = tqdm(self.dataloader)
            running_results = {'batch_sizes': 0, 'loss': 0}
            ite = 0
            for module in teacher_model.children():
                module.train(True)
            for input, target in train_bar:  # enumerate(self.dataloader):
                if target.size(1) == 3:
                    ite += 1
                    lr_patch = Variable(input, requires_grad=False).cuda(gpuID)
                    hr_patch = Variable(target, requires_grad=False).cuda(gpuID)

                    batch_size = input.size(0)
                    running_results['batch_sizes'] += batch_size

                    unloader = transforms.ToPILImage()
                    image = input.cpu().clone()
                    image = image.squeeze(0)
                    haze = unloader(image)
                    x = cv.cvtColor(np.asarray(haze), cv.COLOR_RGB2BGR)
                    pre_haze = prepro(x).cuda(gpuID)

                    outs = []

                    fea1, fea2, fea3, fea4, fea5, fea6 = meta(pre_haze)
                    outs.append(fea1)
                    outs.append(fea2)
                    outs.append(fea3)
                    outs.append(fea4)
                    outs.append(fea5)

                    haze_features = cat((fea1, fea2, fea3, fea4, fea5), 1)

                    rec_hazy_free, student_yy, student_yyy, student_yyyy, student_yyyyy, student_yyyyyy = teacher_model.forward(lr_patch, haze_features, pre_haze)

                    rec_hazy_free = rec_hazy_free.cuda()

                    dimage = rec_hazy_free.cpu().clone()
                    dimage = dimage.squeeze(0)
                    dhaze = unloader(dimage)
                    dx = cv2.cvtColor(np.asarray(dhaze), cv2.COLOR_RGB2BGR)
                    pre_dhaze = prepro(dx).to(device)
                    de_f1, de_f2, de_f3, de_f4, de_f5, de_f6 = meta(pre_dhaze)

                    simage = target.cpu().clone()
                    simage = simage.squeeze(0)
                    src = unloader(simage)
                    src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
                    pre_src = prepro(src).to(device)
                    src_f1, src_f2, src_f3, src_f4, src_f5, src_f6 = meta(pre_src)

                    optimizer.zero_grad()
                    l1_loss = L1_loss(rec_hazy_free, hr_patch)
                    perceptual_loss = loss_network(rec_hazy_free, hr_patch)
                    fe_loss = (
                            F.mse_loss(de_f1, src_f1) + F.mse_loss(de_f2, src_f2) + F.mse_loss(de_f3, src_f3) +
                            F.mse_loss(de_f4, src_f4) + F.mse_loss(de_f5, src_f5))  # + F.mse_loss(de_f6, src_f6)

                    # out, student_yy1, student_yyy1, student_yyyy1, student_yyyyy1, student_yyyyyy1 = student(lr_patch, haze_features, pre_haze)

                    # out = out.cuda()
                    target1 = target.cuda()
                    loss = ContrastLoss(ablation=False)
                    # 将输入数据传递给 forward 方法计算损失
                    # cont = loss(rec_hazy_free, target1, out)

                    unloader = transforms.ToPILImage()
                    image1 = target.cpu().clone()
                    image2 = image1.squeeze(0)
                    g_t = unloader(image2)
                    x = cv.cvtColor(np.asarray(g_t), cv.COLOR_RGB2BGR)
                    pre_gt = prepro(x)
                    pre_gt = pre_gt.to(device)
                    fea1, fea2, fea3, fea4, fea5, fea6 = model(pre_gt)
                    gt_features = cat((fea1, fea2, fea3, fea4, fea5), 1)

                    rec_gt, teacher_yy, teacher_yyy, teacher_yyyy, teacher_yyyyy, teacher_yyyyyy = teacher.forward(hr_patch, gt_features, pre_gt)

                    kt_loss = nn.L1Loss()
                    teacher_recons_loss0 = kt_loss(teacher_yy, student_yy)
                    teacher_recons_loss1 = kt_loss(teacher_yyy, student_yyy)
                    teacher_recons_loss2 = kt_loss(teacher_yyyy, student_yyyy)
                    teacher_recons_loss3 = kt_loss(teacher_yyyyy, student_yyyyy)
                    teacher_recons_loss4 = kt_loss(teacher_yyyyyy, student_yyyyyy)
                    teacher_recons_loss = 0.2 * teacher_recons_loss0 + 0.2 * teacher_recons_loss1 + 0.2 * teacher_recons_loss2 + 0.2 * teacher_recons_loss3 + 0.2 * teacher_recons_loss4

                    caploss = CAPLoss(rec_hazy_free)
                    labloss = LabLoss(rec_hazy_free,target)
                    loss = l1_loss + 0.04 * perceptual_loss + 0.04 * fe_loss + 0.001*caploss + 0.0001*labloss + teacher_recons_loss
                    #+ 0.04 * fe_loss + 0.04*cont+ 0.001*caploss + 0.0001*labloss+ 0.04*cont
                    meta_optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    meta_optimizer.step()
                    train_loss += loss.item()
                    running_results['loss'] += loss.item()
                    train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                        epoch, 100, running_results['loss'] / running_results['batch_sizes']))
                    if ite % 80 == 0:
                        # print(output)
                        vutils.save_image(lr_patch.data, 'D:/AProject/meta/train_mata/student_try_result/input_result/input_sample%d.png' % gpuID, normalize=True)
                        vutils.save_image(hr_patch.data, 'D:/AProject/meta/train_mata/student_try_result/target_result/target_sample%d.png' % gpuID, normalize=True)  # True
                        vutils.save_image(rec_hazy_free.data, 'D:/AProject/meta/train_mata/student_try_result/output_result/output_sample%d.png' % gpuID, normalize=True)
                        # print(output)

            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('time ', time.time() - start_time)
            torch.save(teacher_model.state_dict(), 'D:/AProject/meta/train_mata/student_try_model/student_model_%d.pth' % int(epoch))
            torch.save(meta.state_dict(),'D:/AProject/meta/train_mata/student_try_model/meta/meta_model_%d.pth' % int(epoch))
            # check val/test performance
            if epoch % test_interval == 0:
                with torch.no_grad():
                    print('------------------------')
                    for module in teacher_model.children():
                        module.train(False)
                    test_psnr = 0
                    test_ssim = 0
                    eps = 1e-10
                    test_ite = 0
                    # for image_name,input, target in enumerate(self.val_loader):
                    for i, (input, target) in enumerate(self.val_loader):
                        target = target.cuda()
                        lr_patch = Variable(input, requires_grad=False).cuda(gpuID)
                        hr_patch = Variable(target, requires_grad=False).cuda(gpuID)

                        unloader = transforms.ToPILImage()
                        image = input.cpu().clone()
                        image = image.squeeze(0)
                        haze = unloader(image)
                        x = cv.cvtColor(np.asarray(haze), cv.COLOR_RGB2BGR)
                        pre_haze = prepro(x).cuda(gpuID)

                        outs = []

                        fea1, fea2, fea3, fea4, fea5, fea6 = meta(pre_haze)
                        outs.append(fea1)
                        outs.append(fea2)
                        outs.append(fea3)
                        outs.append(fea4)
                        outs.append(fea5)

                        haze_features = cat((fea1, fea2, fea3, fea4, fea5), 1)

                        rec_hazy_free, student_yy, student_yyy, student_yyyy, student_yyyyy, student_yyyyyy = teacher_model.forward(lr_patch, haze_features, pre_haze)
                        # output = model(lr_patch,haze_features )
                        # save images
                        vutils.save_image(rec_hazy_free.data, 'D:/AProject/meta/train_mata/student_try_result_t/output_result/%05d.png' % (int(i)), padding=0,
                                          normalize=True)  # False
                        vutils.save_image(lr_patch.data, 'D:/AProject/meta/train_mata/student_try_result_t/input_result/%05d.png' % (int(i)), padding=0,
                                          normalize=True)
                        vutils.save_image(hr_patch.data, 'D:/AProject/meta/train_mata/student_try_result_t/target_result/%05d.png' % (int(i)), padding=0,
                                          normalize=True)
                        # Calculation of SSIM and PSNR values
                        # print(output)
                        output = rec_hazy_free.data.cpu().numpy()[0]
                        output[output > 1] = 1
                        output[output < 0] = 0
                        output = output.transpose((1, 2, 0))
                        hr_patch = hr_patch.data.cpu().numpy()[0]
                        hr_patch[hr_patch > 1] = 1
                        hr_patch[hr_patch < 0] = 0
                        hr_patch = hr_patch.transpose((1, 2, 0))
                        # SSIM
                        # test_ssim += ski_ssim(output, hr_patch, win_size=5, data_range=1, multichannel=True)

                        test_ssim = test_ssim + ssim(rec_hazy_free, target)

                        # test_ssim = test_ssim + ssim(output, target)
                        # PSNR
                        imdf = (output - hr_patch) ** 2
                        mse = np.mean(imdf) + eps
                        test_psnr += 10 * math.log10(1.0 / mse)
                        test_ite += 1
                    test_psnr /= (test_ite)
                    test_ssim /= (test_ite)
                    print('Valid PSNR: {:.4f}'.format(test_psnr))
                    print('Valid SSIM: {:.4f}'.format(test_ssim))
                    #写入每轮的PSNR和SSIM
                    f = open('PSNR.txt', 'a')
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([epoch, test_psnr, test_ssim])
                    f.close()
                    print('------------------------')
                torch.save(teacher_model.state_dict(), 'D:/AProject/meta/train_mata/student_try_model/teacher_model_t1_%d.pth' % int(epoch))

        return train_loss
# out1:

# 设置PyTorch线程数


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    parser = argparse.ArgumentParser(description='Operation-wise Attention Network')
    parser.add_argument('--gpu_num', '-g', type=int, default=0, help='Num. of GPUs')
    parser.add_argument('--mode', '-m', default='yourdata', help='Mode (mix / yourdata)')
    args = parser.parse_args()

    # --- Optimization of the CNN architecture ---
    if args.mode == 'mix':
        cnn = CNN_train(args.mode, imgSize=63, batchsize=32)
        acc = cnn(None, 0, epoch_num=150, gpu_num=args.gpu_num)
    elif args.mode == 'yourdata':
        cnn = CNN_train(args.mode, imgSize=256, batchsize=1)#256 2
        acc = cnn(None, 0, epoch_num=100, gpu_num=args.gpu_num)

