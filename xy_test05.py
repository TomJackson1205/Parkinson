import pandas as pd

data=pd.read_csv(r"test\pd_speech_features.csv")
print(data.shape)
names=['GNE_NSR_TKEO','maxIntensity',"VFER_entropy",'numPeriodsPulses','locAbsJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter'
        ,'locDbShimmer','ddaShimmer', 'apq3Shimmer', 'apq5Shimmer','apq11Shimmer','meanIntensity',"IMF_NSR_entropy"]
for name in names:
    data_new=data.drop([name],axis=1)
    data=data_new
print(data.shape)#(756, 740)
data_new.to_csv("test/train_1.csv",index=0)