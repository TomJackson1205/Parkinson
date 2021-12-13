#TQWT数据进行归一化，生成归一化后的训练文件
import numpy as np
from numba import jit


@jit
def Normalized(datas,min_Num,denominator):
    js,ks=datas.shape
    for j in range(js):
        for k in range(ks):
            datas[j][k] = ((float(datas[j][k]) - min_Num) / denominator)
    return datas

#输入Concat数据，返回归一化后的Concat
def Concat_Normalized(Concat):
    '''
    # 1列
    PPE = Concat[:, 0]
    DFA = Concat[:, 1]
    RPDE = Concat[:, 2]
    numPulses = Concat[:, 3]
    numPeriods_Pulses = Concat[:, 4]
    meanHarmtoNoise = Concat[:, 20]
    GNE_SNR_SEO = Concat[:, 38]
    VFER_entropy = Concat[:, 43]
    IMF_SNR_entropy = Concat[:, 50]
    # 2列
    Fundermaental_frequency_parameters = Concat[:, 5:7]
    Harmonicity_parameters = Concat[:, 18:20]
    GNE_NSR_SEO = Concat[:, 39:41]
    VFER_std = Concat[:, 41:43]
    VFER_SNR_SEO = Concat[:, 44:46]
    VFER_NSR_SEO = Concat[:, 46:48]
    IMF_SNR_TEKO = Concat[:, 48:50]
    # 3列
    Intensity_Parameters = Concat[:, 21:24]
    GQ = Concat[:, 32:35]
    GNE_std = Concat[:, 35:38]
    IMF_NSR_entropy = Concat[:, 51:54]
    # 其他
    Formant_Frequeneies = Concat[:, 24:28]
    Bandewith_Paramenters = Concat[:, 28:32]

    Jitter_variants = Concat[:, 7:12]
    Shimmer_variants = Concat[:, 12:18]'''
    for i in [0,1,2,3,4,20,38,43,50]:
        clumn=Concat[:,i]
        # 计算第i列的最大最小值
        max_Num = np.amax(clumn)
        min_Num = np.amin(clumn)
        denominator = max_Num - min_Num
        # 最大最小归一化
        clumn = np.array([((float(x) - min_Num) / denominator) for x in clumn])
        # 用归一化后的数据替换原数据
        Concat[:, i] = clumn

    for j in [5,18,39,41,44,46,48]:
        clumn2 = Concat[:, j:j+2]
        max_Num = np.amax(clumn2)
        min_Num = np.amin(clumn2)
        denominator = max_Num - min_Num
        Concat[:, j:j + 2]=Normalized(clumn2,min_Num,denominator)

    for k in [21,32,35,51]:
        clumn3 = Concat[:, k:k+3]
        max_Num = np.amax(clumn3)
        min_Num = np.amin(clumn3)
        denominator = max_Num - min_Num
        Concat[:, k:k+3]=Normalized(clumn3,min_Num,denominator)

    for l in [24,28]:
        clumn4 = Concat[:, l:l+4]
        max_Num = np.amax(clumn4)
        min_Num = np.amin(clumn4)
        denominator = max_Num - min_Num
        Concat[:, l:l+4]=Normalized(clumn4,min_Num,denominator)

    Formant_Frequeneies,Shimmer_variants = Concat[:, 7:12],Concat[:, 12:18]
    max_Num,max_Num_1 = np.amax(Formant_Frequeneies),np.amax(Shimmer_variants)
    min_Num,min_Num_1 = np.amin(Formant_Frequeneies),np.amin(Shimmer_variants)
    denominator,denominator_1 = max_Num - min_Num,max_Num_1-min_Num_1

    Concat[:, 7:12] = Normalized(Formant_Frequeneies, min_Num, denominator)
    Concat[:, 12:18]=Normalized(Shimmer_variants, min_Num_1, denominator_1)
    return Concat

#输入MFCC数据，返回归一化后的MFCC
def MFCC_Normalized(MFCC):
    '''
    mean_Log_energy=MFCC[:,0]
    mean_MFCC_0th_coef_12th=MFCC[:,1:14]
    mean_delta_log_energy=MFCC[:,14]
    mean_0th_delta_12th=MFCC[:,15:28]
    mean_delta_delta_log_energy=MFCC[:,28]
    mean_0th_delta_delta_12th=MFCC[:,29:42]
    std_Log_energy=MFCC[:,42]
    std_MFCC_0th_coef_12th=MFCC[:,43:56]
    std_delta_log_energy=MFCC[:,56]
    std_0th_delta_12th=MFCC[:,57:70]
    std_delta_delta_log_energy=MFCC[:,70]
    std_0th_delta_delta_12th=MFCC[:,71:84]
    '''
    for i in [0,14,28,42,56,70]:
        log_energy = MFCC[:,i]
        MFCC_0th_12th=MFCC[:,i+1:i+14]
        max_Num_log,max_Num_0th_12th = np.amax(log_energy),np.amax(MFCC_0th_12th)
        min_Num_log,min_Num_0th_12th = np.amin(log_energy),np.min(MFCC_0th_12th)
        denominator_log,denominator_0th_12th = max_Num_log - min_Num_log,max_Num_0th_12th-min_Num_0th_12th
        # 最大最小归一化
        log_energy = np.array([((float(x) - min_Num_log) / denominator_log) for x in log_energy])
        # 用归一化后的数据替换原数据
        MFCC[:, i] = log_energy
        MFCC[:, i + 1:i + 14]=Normalized(MFCC_0th_12th,min_Num_0th_12th,denominator_0th_12th)

    return MFCC

#输入Wavelet数据，返回归一化后的Wavelet
def Wavelet_Normalized(Wavelet,Condition=True):
    '''
    Ea=Wavelet[:,0]
    Ed_1_coef_10=Wavelet[:,1:11]
    det_entropy_shannon_1_coef_10=Wavelet[:,11:21]
    det_entropy_log_1_coef_10=Wavelet[:,21:31]
    det_TKEO_mean_1_coef_10 = Wavelet[:, 31:41]
    det_TKEO_std_1_coef_10 = Wavelet[:, 41:51]
    app_entropy_shannon_1_coef_10 = Wavelet[:, 51:61]
    app_entropy_log_1_coef_10 = Wavelet[:, 61:71]
    app_det_TKEO_mean_1_coef_10 = Wavelet[:, 71:81]
    app_TKEO_std_1_coef_10 = Wavelet[:, 81:91]
    Ea2=Wavelet[:,91]
    Ed2_1_coef_10=Wavelet[:,92:102]
    det_LT_entropy_shannon_1_coef_10 = Wavelet[:, 102:112]
    det_LT_entropy_log_1_coef_10 = Wavelet[:, 112:122]
    det_LT_TKEO_mean_1_coef_10 = Wavelet[:, 122:132]
    det_LT_TKEO_std_1_coef_10 = Wavelet[:, 132:142]
    app_LT_entropy_shannon_1_coef_10 = Wavelet[:, 142:152]
    app_LT_entropy_log_1_coef_10 = Wavelet[:, 152:162]
    app_LT_det_TKEO_mean_1_coef_10 = Wavelet[:, 162:172]
    app_LT_TKEO_std_1_coef_10 = Wavelet[:, 172:182]
'''
    if Condition:
        #单独处理第0列(Ea)和第91列(Ea2)的数据
        for i in [0,91]:
            #取出数组中第i列的所有数据
            Ea=Wavelet[:,i]
            #计算第i列的最大最小值
            max_Num = np.amax(Ea)
            min_Num = np.amin(Ea)
            denominator = max_Num - min_Num
            #最大最小归一化
            Ea =np.array([((float(x) - min_Num) / denominator) for x in Ea])
            #用归一化后的数据替换原数据
            Wavelet[:,i]=Ea
        #定义切片的左右端点，数学区间表达式为：[left,right)
        left=0
        right=0
        for i in range(1, 19):
            if i <10:
                left=10 * (i-1)+1
                right=10 * i + 1
            else:
                left = 10 * (i - 1) + 2
                right = 10 * i + 2
            wavelet_data = Wavelet[:, left:right]
            # 计算第i段特征的最大最小值
            max_Num = np.amax(wavelet_data)
            min_Num = np.amin(wavelet_data)
            denominator = max_Num - min_Num
            # 第i段特征的归一化
            Wavelet[:, left:right]=Normalized(wavelet_data,min_Num,denominator)
    #两段合起来做归一化
    else:
        Ea_datas = Wavelet[:, :91]
        Ea2_datas = Wavelet[:, 91:]
        for i in range(10):
            if i == 0:
                Ea = Ea_datas[:, i]
                Ea2 = Ea2_datas[:, i]
                max_Num = max(np.amax(Ea), np.amax(Ea2))
                min_Num = min(np.amin(Ea), np.amin(Ea2))
                denominator = max_Num - min_Num
                Ea_ = np.array([((float(x) - min_Num) / denominator) for x in Ea])
                Ea2_ = np.array([((float(x) - min_Num) / denominator) for x in Ea2])
                Ea_datas[:, i] = Ea_
                Ea2_datas[:, i] = Ea2_
            else:
                Ea = Ea_datas[:, 10 * (i - 1) + 1:10 * i + 1]
                Ea2 = Ea2_datas[:, 10 * (i - 1) + 1:10 * i + 1]
                # 计算第i段特征的最大最小值
                max_Num = max(np.amax(Ea), np.amax(Ea2))
                min_Num = min(np.amin(Ea), np.amin(Ea2))
                denominator = max_Num - min_Num
                # 第i段特征的归一化
                Ea_datas[:, 10 * (i - 1) + 1:10 * i + 1] = Normalized(Ea,min_Num,denominator)
                Ea2_datas[:, 10 * (i - 1) + 1:10 * i + 1] = Normalized(Ea2,min_Num,denominator)
        Wavelet[:, :91] = Ea_datas
        Wavelet[:, 91:] = Ea2_datas
    return Wavelet

#输入TQWT数据，返回归一化后的TQWT
def TQWT_Normalized(TQWT):
    '''
    tqwt_energy_dec_1_36=TQWT[:,:36]
    tqwt_entropy_shannon_dec_1_36=TQWT[:,36:72]
    tqwt_entropy_log_dec_1_36=TQWT[:,72:108]
    tqwt_TKEO_mean_dec_1_36=TQWT[:,108:144]
    tqwt_TKEO_std_dec_1_36=TQWT[:,144:180]
    tqwt_medianValue_dec_1_36=TQWT[:,180:216]
    tqwt_meanValue_dec_1_36=TQWT[:,216:252]
    tqwt_stdValue_dec_1_36=TQWT[:,252:288]
    tqwt_minValue_dec_1_36=TQWT[:,288:324]
    tqwt_maxValue_dec_1_36=TQWT[:,324:360]
    tqwt_skewnessValue_dec_1_36=TQWT[:,360:396]
    tqwt_kurtosisValue_dec_1_36=TQWT[:,396:432]
    '''
    for i in range(0, 12):
        tqwt_data = TQWT[:, 36 * i:36 * (i + 1)]
        # 计算第i段特征的最大最小值
        max_Num = np.amax(tqwt_data)
        min_Num = np.amin(tqwt_data)
        denominator = max_Num - min_Num
        TQWT[:, 36 * i:36 * (i + 1)] = Normalized(tqwt_data,min_Num,denominator)

    return TQWT


if __name__ == '__main__':

    with open("train_data\pd_speech_features2.csv") as f:
        lines=f.readlines()
        all_datas=[]
        count=0
        #将文件中所有数据放入一个列表
        for line in lines:
            count+=1
            strs = line.split(",")
            datas=[float(x) for x in strs[0:755]]

            all_datas.append(datas)
        #将列表转换为矩阵，便于后面的矩阵操作
        all_datas=np.array(all_datas)
        # Concat最大最小归一化
        Concat = all_datas[:, 2:56]
        all_datas[:, 2:56] = Concat_Normalized(Concat)
        # MFCC最大最小归一化
        MFCC = all_datas[:, 56:140]
        all_datas[:, 56:140] = MFCC_Normalized(MFCC)
        #Wavelet最大最小归一化
        Wavelet = all_datas[:,140:322]
        all_datas[:,140:322]=Wavelet_Normalized(Wavelet)
        # TQWT最大最小归一化
        TQWT = all_datas[:, 322:754]
        all_datas[:, 322:754] = TQWT_Normalized(TQWT)
        # 生成归一化后的训练集
        with open("train_data/name_train.csv","w") as tf:
            for data in all_datas:
                tf.write(",".join([str(a) for a in data]) + '\n')
        tf.close()
    f.close()

