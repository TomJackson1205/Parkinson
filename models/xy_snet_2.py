import torch.nn as nn
import torch


class Main_net(nn.Module):
    def __init__(self):
        super(Main_net,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.linear4=nn.Linear(512,2)
    def forward(self,x):
        conv_1=self.conv1(x)
        # print(conv_1.shape)#2,3,6,14
        y = conv_1.reshape(conv_1.size()[0], -1)
        out = self.linear4(y)
        return out

# if __name__ == '__main__':
#     x=torch.Tensor(1,1,54)#54,84,182,432
#     net=Main_net()
#     out=net(x)
#     print(out.shape)