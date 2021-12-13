#-*- coding: utf-8 -*-
import torch
import os
import time

from models.TQWT_net import MainNet
# from models.xy_snet_1 import Main_net as MainNet
from utils.TQWT_datasets import Mydataset
from split_data import split_csv
from yyr_test1 import main
from matplotlib import pyplot as plt


def train():
    # Set Dataloader
    train_Dataset = Mydataset(r"./train_data/TQWT_train.csv")
    # cudaあればcudaを使います。なかったらcpuを使います。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    net = MainNet().to(device=device)

    opt = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=512, shuffle=True)
    # plt.ion()
    # plt.title("TQWT_loss")
    # color=["red","blue"]
    # losses = []
    for i in range(200):#訓練回数２００回
        plt.clf()
        # point_l = []
        # point_r = []
        # lables=[]

        for id, data, lable in train_loader:
            # inference
            out = net(data.to(device))
            # print(out)
            loss = loss_fn(out, lable.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        # losses.append(loss.item())
            # out=out.detach().cpu().numpy()
            # point_l.extend(out[:,0])
            # point_r.extend(out[:, 1])
            # lables.extend(lable)
        print("epoch:{0} loss:{1}".format(i, loss.item()))
        # plt.plot(losses)
        # for j in range(len(lables)):
        #     plt.scatter(point_l[j],point_r[j],color=color[lables[j]])
    #     plt.pause(0.1)
    # plt.ioff()
    # plt.savefig("TQWT_loss_1.png")
    # plt.show()
    torch.save(net.state_dict(), r'./weight/TQWT.pt')

def test():
    # Set Dataloader
    test_Dataset = Mydataset(r"./train_data/TQWT_test.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    net = MainNet().to(device=device)
    if os.path.exists(r'./weight/TQWT.pt'):
        net.load_state_dict(torch.load(r'./weight/TQWT.pt'))
    net.eval()
    # Run inference
    test_loader = torch.utils.data.DataLoader(test_Dataset, batch_size=3, shuffle=False)
    for j, (id, data, lable) in enumerate(test_loader):
        # inference
        out = net(data.to(device))
        result = torch.argmax(out, dim=1).cpu().numpy().tolist()

        pre_lable = max(result, key=result.count)

        lable = lable.numpy().tolist()
        tre_lable = max(lable, key=lable.count)
    # Save predict results
    # 予測した結果をpre_Concat.txtに保存します。yyr_test1ファイルで評価標準の計算に使います。
    with open(r"result/my_net/table_3/pre_MFCC_Wavelet_Concat.txt", "a") as f:
        f.write(str(int(id[0][0])) + "," + str(pre_lable))
        f.write("\n")
    f.close()
    # Save true results
    # 元データファイルより、パーキンソン病であるかどうかのラベルを抽出し、true_Concat.txtに書き込みます。
    with open(r"result/my_net/table_3/true_MFCC_Wavelet_Concat.txt", "a") as ft:
        ft.write(str(int(id[0][0])) + "," + str(tre_lable))
        ft.write("\n")
    ft.close()

    if os.path.exists(r'./weight/TQWT.pt'):
        os.remove(r'./weight/TQWT.pt')


if __name__ == '__main__':
    path = r"./train_data/train_1.csv"
    train_path=r"./train_data/TQWT_train.csv"
    test_path=r"./train_data/TQWT_test.csv"
    start_time = time.time()
    for i in range(0, 252):
        print("person:{0}".format(i))
        split_csv(path, 3 * i, train_path, test_path)
        print("start train...")
        train()
        print("start test...")
        test()
    end_time = time.time()
    print("training time：{0}".format(end_time - start_time))
    # yyr_test1ファイル（146行-149行）で評価標準の計算に使います。
    path1=r"result/my_net/table_3/pre_MFCC_Wavelet_Concat.txt"
    # yyr_test1ファイル（146行-149行）で評価標準の計算に使います。
    path2 = r"result/my_net/table_3/true_MFCC_Wavelet_Concat.txt"
    main(path1,path2)