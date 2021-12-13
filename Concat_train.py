#-*- coding: utf-8 -*-
import torch
import os
import time

from models.Concat_net import Main_net as MainNet
# from models.xy_snet_2 import Main_net as MainNet
from utils.Concat_datasets import Mydataset
from split_data import split_csv
from yyr_test1 import main


def train():
    # Set Dataloader
    train_Dataset = Mydataset(r"./train_data/Concat_train.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    net = MainNet().to(device=device)
    # Run inference
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
    # weightを保存、テスト部分に使います。weightパラメータは多すぎなので、メモリを考えて、第７3行で削除します。
    torch.save(net.state_dict(), r'./weight/Concat.pt')

def test():
    # Set Dataloader
    test_Dataset = Mydataset(r"./train_data/Concat_test.csv")
    # cudaあればcudaを使います。なかったらcpuを使います。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    net = MainNet().to(device=device)
    if os.path.exists(r'./weight/Concat.pt'):#保存されたweightファイルをアップロードします
        net.load_state_dict(torch.load(r'./weight/Concat.pt'))
    net.eval()
    # Run inference
    test_loader = torch.utils.data.DataLoader(test_Dataset, batch_size=3, shuffle=False)
    for j, (id, data, lable) in enumerate(test_loader):
        # inference
        out = net(data.to(device))
        result =torch.argmax(out, dim=1).cpu().numpy().tolist()

        pre_lable= max(result, key=result.count)

        lable=lable.numpy().tolist()
        tre_lable=max(lable, key=lable.count)

    # Save predict results
    with open(r"./result/my_net/table_3/pre_TQWT_MFCC_Concat.txt", "a") as f:
        f.write(str(int(id[0][0])) + "," + str(pre_lable))
        f.write("\n")
    f.close()
    # Save true results
    with open(r"./result/my_net/table_3/true_TQWT_MFCC_Concat.txt", "a") as ft:
        ft.write(str(int(id[0][0])) + "," + str(tre_lable))
        ft.write("\n")
    ft.close()
    if os.path.exists(r'./weight/Concat.pt'):
        os.remove(r'./weight/Concat.pt')#Concat.ptを削除します


if __name__ == '__main__':
    path = r"./train_data/train_1.csv"
    train_path=r"./train_data/Concat_train.csv"
    test_path=r"./train_data/Concat_test.csv"
    start_time = time.time()
    for i in range(163, 252):
        print("person:{0}".format(i))
        split_csv(path, 3 * i, train_path, test_path)
        print("start train...")
        train()
        print("start test...")
        test()
    end_time = time.time()
    print("training time：{0}".format(end_time - start_time))
    path1=r"./result/my_net/table_3/pre_TQWT_MFCC_Concat.txt"
    path2 = r"./result/my_net/table_3/true_TQWT_MFCC_Concat.txt"
    main(path1,path2)


