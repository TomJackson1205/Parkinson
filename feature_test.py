#-*- coding: utf-8 -*-
import torch

# from models.Concat_net import MainNet
from models.xy_snet_1 import Main_net as MainNet
from utils.TQWT_datasets import Mydataset
from matplotlib import pyplot as plt
from center_loss import CenterLoss


def train():
    c = ['#ff0000','#00ff00']
    # Set Dataloader
    train_Dataset = Mydataset(r"./train_data/train.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    net = MainNet().to(device=device)
    # Run inference
    opt = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    # center_loss_fn=CenterLoss(2,2).to(device=device)
    batch=512
    epoches = 150
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch, shuffle=True)
    plt.ion()

    for epoch in range(epoches):#訓練回数２００回
        features=[]
        lables=[]
        plt.clf()
        for id, data, lable in train_loader:
            # inference
            feature,out = net(data.to(device))
            # center_loss = center_loss_fn(feature, lable.to(device))
            loss = loss_fn(out, lable.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            features.append(feature)
            lables.append(lable)
        features_ = torch.cat(features, 0).detach().cpu().numpy()
        lables_ =torch.cat(lables,0).detach().cpu().numpy()
        plt.legend(['0', '1'], loc='upper right')
        for i in range(2):
            plt.plot(features_[lables_ == i, 0], features_[lables_ == i, 1], '.', c=c[i])
        plt.pause(0.1)
        plt.title("epoch={}".format(epoch))
        print("epoch:{0} loss:{1}".format(epoch,loss.item()))
    plt.savefig('相关性图/{}_{}.jpg'.format(epoches,batch))
    plt.ioff()



if __name__ == '__main__':
    train()


