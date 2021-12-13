#-*- coding: utf-8 -*-
'''
convolution layer は同じです。違うfeature setsよりdense layerの入力が違います。それによって、feature setsごとnet.pyが必要です
また、feature setsごとnet.pyを作れば、4つ特徴セットが同時にトレーニングすることができます。
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

class MainNet(nn.Module):
    def __init__(self):
        super(MainNet,self).__init__()
        # convolution layer
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=7,stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7,stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,1),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2,1),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3,stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,1),
        )
        # dense layer
        self.leaner=nn.Sequential(
            nn.Linear(1824,32),
            nn.ReLU(),

            nn.Linear(32,2)
        )

    def forward(self,x):
        conv=self.conv1(x)
        y=conv.reshape(conv.size()[0],-1)
        out=self.leaner(y)

        # outs=torch.softmax(out,dim=1)

        return out
# if __name__ == '__main__':
#     net=MainNet()
#     x=torch.Tensor(1,1,84)
#     net(x)