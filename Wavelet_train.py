#-*- coding: utf-8 -*-
import torch
import os
import time

from models.Wavelet_net import MainNet
from utils.Wavelet_datasets import Mydataset
from split_data import split_csv
from yyr_test1 import main


def train():
    # Set Dataloader
    train_Dataset = Mydataset(r"./train_data/Wavelet_train.csv")
    # cudaあればcudaを使います。なかったらcpuを使います。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    net = MainNet().to(device=device)
    # opt = torch.optim.Adam(net.parameters(), weight_decay=0.01)
    opt = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=512, shuffle=True)
    for i in range(200):#訓練回数２００回
        for id, data, lable in train_loader:
            # inference
            out = net(data.to(device))
            loss = loss_fn(out, lable.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("epoch:{0} loss:{1}".format(i,loss.item()))
    torch.save(net.state_dict(), r'./weight/Wavelet.pt')

def test():
    # Set Dataloader
    test_Dataset = Mydataset(r"./train_data/Wavelet_test.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    net = MainNet().to(device=device)
    if os.path.exists(r'./weight/Wavelet.pt'):
        net.load_state_dict(torch.load(r'./weight/Wavelet.pt'))
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
    with open(r"result/my_net/table_3/pre_TQWT_Wavelet_Concat.txt", "a") as f:
        f.write(str(int(id[0][0])) + "," + str(pre_lable))
        f.write("\n")
    f.close()
    # Save true results
    # 元データファイルより、パーキンソン病であるかどうかのラベルを抽出し、true_Concat.txtに書き込みます。
    with open(r"result/my_net/table_3/true_TQWT_Wavelet_Concat.txt", "a") as ft:
        ft.write(str(int(id[0][0])) + "," + str(tre_lable))
        ft.write("\n")
    ft.close()

    if os.path.exists(r'./weight/Wavelet.pt'):
        os.remove(r'./weight/Wavelet.pt')


if __name__ == '__main__':
    path = r"./train_data/train_1.csv"
    train_path=r"./train_data/Wavelet_train.csv"
    test_path=r"./train_data/Wavelet_test.csv"
    start_time = time.time()
    for i in range(0,252):
        print("person:{0}".format(i))
        split_csv(path, 3 * i,train_path,test_path)
        print("start train...")
        train()
        print("start test...")
        test()
    end_time = time.time()
    print("training time：{0}".format(end_time - start_time))
    # yyr_test1ファイル（146行-149行）で評価標準の計算に使います。
    path1=r"result/my_net/table_3/pre_TQWT_Wavelet_Concat.txt"
    # yyr_test1ファイル（146行-149行）で評価標準の計算に使います。
    path2 = r"result/my_net/table_3/true_TQWT_Wavelet_Concat.txt"
    main(path1,path2)