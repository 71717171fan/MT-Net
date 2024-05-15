import torch.nn as nn
import torch
from model200314 import *

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

#此代码建立一个教师帮助学生学习的结构
class CKTModule(nn.Module):
    def __init__(self, n_teachers):
        super(CKTModule, self).__init__()
        self.teachers = torch.load('C:/Users/86198/PycharmProjects/pythonProject/myself/meta/train_mata/teacher_model/teacher_model.pth')
        self.student = TransformNet(32).to(device)

    def forward(self, x):
        teacher_outputs = self.teacher(x)
        student_output = self.student(x)

        return teacher_outputs, student_output