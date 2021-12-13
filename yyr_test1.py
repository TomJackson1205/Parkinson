#-*- coding: utf-8 -*-

import numpy as np
import math

def getLabelData(file_dir):
    labels = []
    with open(file_dir,'r',encoding="utf-8") as f:
        for i in f.readlines():
            labels.append(i.strip().split(',')[1])
    return labels

def getLabel2idx(labels):

    label2idx = dict()
    for i in labels:
        if i not in label2idx:
            label2idx[i] = len(label2idx)
    return label2idx


def buildConfusionMatrix(predict_file,true_file):
    '''
    confusion matrixを生成
    戻り値: returns the matrix numpy

    '''
    true_labels = getLabelData(true_file)
    predict_labels = getLabelData(predict_file)
    label2idx = getLabel2idx(true_labels)
    confMatrix = np.zeros([len(label2idx),len(label2idx)],dtype=np.int32)
    for i in range(len(true_labels)):
        true_labels_idx = label2idx[true_labels[i]]
        predict_labels_idx = label2idx[predict_labels[i]]
        confMatrix[true_labels_idx][predict_labels_idx] += 1
    return confMatrix,label2idx



def calculate_all_prediction(confMatrix):

    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100*float(correct_sum)/float(total_sum),2)
    return prediction

def calculate_label_prediction(confMatrix,labelidx):
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100*float(label_correct_sum)/float(label_total_sum),2)
    return prediction

def calculate_label_recall(confMatrix,labelidx):
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100*float(label_correct_sum)/float(label_total_sum),2)
    return recall

def calculate_f1(prediction,recall):
    if (prediction+recall)==0:
        return 0
    return round(2*prediction*recall/(prediction+recall),2)

#MCCを計算します
def MCC(Confusion_matrix):
    TP=Confusion_matrix[0][0]
    FP=Confusion_matrix[0][1]
    FN=Confusion_matrix[1][0]
    TN=Confusion_matrix[1][1]
    N=TN+TP+FN+FP
    S=(TP+FN)/N
    P=(TP+FP)/N
    mcc=(TP/N-S*P)/math.sqrt(P*S*(1-S)*(1-P))
    # M_CC=(TP*TN-FN*FP)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    # # print("M_CC={}".format(M_CC))
    return mcc
def main(predict_file,true_file):
    #Read the file and convert it into confusion matrix, and return label2idx
    confMatrix,label2idx = buildConfusionMatrix(predict_file,true_file)
    total_sum = confMatrix.sum()
    all_prediction = calculate_all_prediction(confMatrix)
    label_prediction = []
    label_recall = []
    print('total_sum=',total_sum,',label_num=',len(label2idx),'\n')
    print("Confusion matrix:","\n")
    for i in label2idx:
        print('    ', i, end=' ')
    print('\n')
    Confusion_matrix=np.array([[0,0],[0,0]])
    for i in label2idx:
        print(i,end=' ')
        label_prediction.append(calculate_label_prediction(confMatrix,label2idx[i]))
        label_recall.append(calculate_label_recall(confMatrix,label2idx[i]))
        for j in label2idx:
            labelidx_i = label2idx[i]
            label2idx_j = label2idx[j]
            Confusion_matrix[labelidx_i][label2idx_j]=confMatrix[labelidx_i][label2idx_j]
            print('  ',confMatrix[labelidx_i][label2idx_j],end=' ')
        print('\n')

    print('prediction(accuracy)=',all_prediction,'%')
    print('individual result\n')
    for ei,i in enumerate(label2idx):
        print(ei,'\t',i,'\t','prediction=',label_prediction[ei],'%,\trecall=',label_recall[ei],'%,\tf1=',calculate_f1(label_prediction[ei],label_recall[ei]))
    p = round(np.array(label_prediction).sum()/len(label_prediction),2)
    r = round(np.array(label_recall).sum()/len(label_prediction),2)
    print("MACRO-averaged:")
    print("prediction={0},recall={1},f1={2},MCC={3}".format(p,r,calculate_f1(p,r),MCC(Confusion_matrix)))

if __name__ == '__main__':
    print("TQWT_MFCC_Wavelet:")
    path1 = r"result/my_net/table_3/pre_TQWT_MFCC_Wavelet.txt"
    path2 = r"result/my_net/table_3/true_TQWT_MFCC_Wavelet.txt"
    main(path1, path2)
    print("\nTQWT_MFCC_Concat")
    path1 = r"./result/my_net/table_3/pre_TQWT_MFCC_Concat.txt"
    path2 = r"./result/my_net/table_3/true_TQWT_MFCC_Concat.txt"
    main(path1, path2)
    print("\nTQWT_Wavelet_Concat")
    path1 = r"result/my_net/table_3/pre_TQWT_Wavelet_Concat.txt"
    path2 = r"result/my_net/table_3/true_TQWT_Wavelet_Concat.txt"
    main(path1, path2)
    print("\nMFCC_Wavelet_Concat")
    path1 = r"result/my_net/table_3/pre_MFCC_Wavelet_Concat.txt"
    path2 = r"result/my_net/table_3/true_MFCC_Wavelet_Concat.txt"
    main(path1, path2)