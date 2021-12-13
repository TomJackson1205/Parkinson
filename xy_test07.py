import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import DataFrame
from scipy.stats import pearsonr

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#删除相关性系数高的列
konusma=pd.read_csv("./test/pd_speech_features.csv")

x=konusma.iloc[:,2:754]
print(x.shape)
x=konusma[konusma.columns[2:754]].corr()
# mask=np.tril(np.ones((x.shape[0],x.shape[0])))
for row in x:
    for column in x:
        if row ==column:
            break
        else:
            if x[row][column]>=0.9:
                del  konusma[row]
                break
print(konusma.shape)
konusma.to_csv("./test/train.csv", index=0)

##################################相关性<0.9,数据长度(756, 432)############################################
# konusma=pd.read_csv("./test/train.csv")
# print(konusma.shape)
# #concat
# plt.figure(figsize=(30,30))
# sns.heatmap(konusma[konusma.columns[2:42]].corr(),annot=True)
# #MFCC
# plt.figure(figsize=(70,70))
# sns.heatmap(konusma[konusma.columns[42:112]].corr(),annot=True)
# #Wavelet
# plt.figure(figsize=(70,70))
# sns.heatmap(konusma[konusma.columns[112:163]].corr(),annot=True)
# #TQWT
# plt.figure(figsize=(300,300))
# sns.heatmap(konusma[konusma.columns[163:431]].corr(),annot=True)
#
# plt.show()
# plt.savefig("./相关性图/test_1.png")

##################################相关性<0.95，数据长度(756, 562)############################################
# konusma=pd.read_csv("./test/train_95.csv")
print(konusma.shape)
# #concat
# plt.figure(figsize=(30,30))
# sns.heatmap(konusma[konusma.columns[2:44]].corr(),annot=True)
#MFCC,start:mean_Log_energy
# plt.figure(figsize=(70,70))
# sns.heatmap(konusma[konusma.columns[44:120]].corr(),annot=True)
#Wavelet,start:Ea
# plt.figure(figsize=(70,70))
# sns.heatmap(konusma[konusma.columns[120:197]].corr(),annot=True)
#TQWT,start:tqwt_energy_dec_1
# plt.figure(figsize=(300,300))
# sns.heatmap(konusma[konusma.columns[197:561]].corr(),annot=True)
#
# plt.show()
# plt.savefig("./相关性图/test_1.png")

##################################相关性<0.8，数据长度(756, 307)############################################
# konusma=pd.read_csv("./test/train_80.csv")
# print(konusma.shape)
# #concat
# plt.figure(figsize=(30,30))
# sns.heatmap(konusma[konusma.columns[2:40]].corr(),annot=True)
# MFCC,start:mean_Log_energy
# plt.figure(figsize=(50,50))
# sns.heatmap(konusma[konusma.columns[40:98]].corr(),annot=True)
# Wavelet,start:Ea
# plt.figure(figsize=(30,30))
# sns.heatmap(konusma[konusma.columns[98:125]].corr(),annot=True)
# TQWT,start:tqwt_energy_dec_1
# plt.figure(figsize=(300,300))
# sns.heatmap(konusma[konusma.columns[322:754]].corr(),annot=True)
# plt.show()
# plt.savefig("./相关性图/test_1.png")

