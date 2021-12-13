#-*- coding: utf-8 -*-　　あるコーディング方法　　
#Concat特徴セット抽出するため
import torch
from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self,path):
        with open(path) as f:
            self.dataset = f.readlines()#行を単位として読み込みます

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        line = self.dataset[index]

        strs = line.strip("\n").split(",")
        id=[float(x) for x in strs[0:1]]#データファイルよりidは第0列

        concat=torch.tensor([float(x) for x in strs[2:56]])
        mfcc=torch.tensor([float(x) for x in strs[56:140]])
        tqwt = torch.tensor([float(x) for x in strs[322:754]])
        datas=torch.cat((concat,mfcc,tqwt),dim=0)
        datas = datas.reshape(1, 570)
        lable=int(float(strs[754]))#パーキンソンかどうかのラベルを所在する列
        return id,datas,lable