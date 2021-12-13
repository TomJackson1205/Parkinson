#画出每个人的每条数据的特征曲线
from matplotlib import pyplot as plt

with open("train_data\pd_speech_features2.csv") as f:
    lines=f.readlines()
    for line in lines:

        plt.ion()
        strs = line.split(",")
        Concat = [float(x) for x in strs[2:56]]
        MFCC = [float(x) for x in strs[56:140]]
        Wavelet = [float(x) for x in strs[140:322]]
        TQWT = [float(x) for x in strs[322:754]]

        plt.plot(TQWT)
        plt.pause(0.5)
        plt.ioff()
        plt.clf()
