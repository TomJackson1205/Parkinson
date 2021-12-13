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
            nn.BatchNorm1d(out_channel)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.1)
        )

    def forward(self,x):
        return self.conv3(self.conv1(x)*self.conv2(x))

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
        self.linear4=nn.Linear(352,2)
    def forward(self,x):
        conv_1=self.conv1(x)
        resnet_2=self.resnet2(conv_1)
        conv_3=self.conv3(resnet_2)
        y = conv_3.reshape(conv_3.size()[0], -1)
        out = self.linear4(y)
        return out

# if __name__ == '__main__':
#     x=torch.Tensor(1,1,698)
#     net=Main_net()
#     out=net(x)
    # print(out.shape)

