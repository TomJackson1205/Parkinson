# #-*- coding: utf-8 -*-
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# class MainNet(nn.Module):
#     def __init__(self):
#         super(MainNet,self).__init__()
#
#         self.conv1=nn.Sequential(
#             nn.Conv1d(in_channels=1,out_channels=32,kernel_size=7,stride=1),#
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7,stride=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1),
#             nn.MaxPool1d(2,1),
#
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,stride=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#
#             nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5,stride=1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.MaxPool1d(2,1),
#             # nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
#
#             nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#
#             nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3,stride=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2,1),
#             # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1),
#         )
#         #dense layer
#         self.leaner=nn.Sequential(
#             nn.Linear(864,2)#960
#         )
#         self.out=nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(2,2)
#         )
#
#     def forward(self,x):
#         conv=self.conv1(x)
#         y=conv.reshape(conv.size()[0],-1)
#         feature=self.leaner(y)
#         out=self.out(feature)
#         return feature,out
#
# if __name__ == '__main__':
#     net=MainNet()
#     x=torch.Tensor(1,1,54)
#     net(x)


import torch.nn as nn
import torch

class Resnet_ibn(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Resnet_ibn,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=2 ,padding=1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.1)
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2 ,padding=1),
            nn.LeakyReLU(0.1),
        )
    def forward(self,x):
        return self.conv1(x)+self.conv2(x)

class Main_net(nn.Module):
    def __init__(self):
        super(Main_net,self).__init__()
        self.conv1=nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.resnet2=nn.Sequential(
            Resnet_ibn(16, 32),
            Resnet_ibn(32, 64),
            Resnet_ibn(64, 128),
            Resnet_ibn(128, 256),
            Resnet_ibn(256, 512)
        )
        self.conv3=nn.Conv1d(in_channels=512,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.linear4=nn.Linear(288,2)
    def forward(self,x):
        conv_1=self.conv1(x)
        resnet_2=self.resnet2(conv_1)
        conv_3=self.conv3(resnet_2)
        y = conv_3.reshape(conv_3.size()[0], -1)
        out = self.linear4(y)
        return out

# if __name__ == '__main__':
#     x=torch.Tensor(1,1,570)
#     net=Main_net()
#     out=net(x)
#     print(out.shape)