#统计最大值和最小值
import os

with open("train_data/train.csv") as f:
    lines=f.readlines()
    datas=[]
    for line in lines:
        strs = line.split(",")
        data = [float(x) for x in strs[2:56]]
        datas.extend(data)

    print(len(datas))
    print("max_Num:",max(datas))
    print("min_Num:",min(datas))