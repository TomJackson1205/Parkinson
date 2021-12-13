#对所有数据进行列归一化
import numpy as np

with open("test/pd_speech_features.csv") as f:
    lines=f.readlines()
    all_datas=[]
    count=0
    #将文件中所有数据放入一个列表
    for line in lines:
        count+=1
        strs = line.split(",")
        print(len(strs))
        datas=[float(x) for x in strs[0:755]]
        all_datas.append(datas)
    #将列表转换为矩阵，便于后面的矩阵操作
    all_datas=np.array(all_datas)
    #按列进行最大最小归一化
    for i in range(3,754):
        #取出数组中第i列的所有数据
        Column=all_datas[:,i:i+1]
        #计算第i列的最大最小值
        max_Num = np.amax(Column)
        min_Num = np.amin(Column)
        denominator = max_Num - min_Num
        print(Column.shape)
        #最大最小归一化
        data5 =np.array([((float(x) - min_Num) / denominator) for x in Column])
        #为归一化后的数据增加一个维度
        data5=data5.reshape(-1,1)
        #用归一化后的数据替换原数据
        all_datas[:,i:i+1]=data5
    #生成归一化后的训练集
    with open("train_data/train_1.csv","w") as tf:
        for data in all_datas:
            tf.write(",".join([str(a) for a in data]) + '\n')
    tf.close()
f.close()

