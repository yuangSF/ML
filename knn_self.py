'''
    function：采用KNN算法，判断分类情况
    input and output：
            datasets：datingTestSet2.txt共4列
            first column：每年获得的飞行常客里程数
            second column：玩视频游戏所耗时间百分比
            third column：每周消费的冰淇淋公升数
            fourth colum：target
    user：lw
    create date:2018-3-18 23:28:40
'''

from numpy import *
import numpy as np
import os
import tkinter.filedialog
import random
from scipy.stats import mode

import matplotlib
import matplotlib.pyplot as plt

# import knn from sklearn
from sklearn import neighbors as knnclass

def calculate_distance(x1,x2,method='Euclid'):
    if method=='Euclid':
        # print(np.sqrt(np.array((x1-x2)*(x1-x2))))
        return np.sqrt(np.array((x1-x2)*(x1-x2)).sum(axis=1))
    elif method=='cos':
        return 1 - np.dot(x1,x2)/(np.linalg.norm(x1)*(np.linalg.norm(x2)))
    else:
        return 3

# to load data from the given path 
def loadData(strFile):
    if strFile=="":
        print("File path must be not null!")
        return []
    elif os.path.exists(strFile):
        # Judgment file in the path
        return np.genfromtxt(strFile)
    else:
        print("There is no file in the path have been showed to me!")

# spit data to train and test set
def trainTestSplit(data,splitRatio=0.7):
    if len(data)==0:
        print("The array is empty!")
    else:
        data = np.array(data)
        [rows,cols] = data.shape
        #random index
        dataIndex = np.array(random.sample(range(rows),rows))
        trainNum = round(rows*splitRatio)
        trainIndex = dataIndex[0:trainNum]
        testIndex = dataIndex[trainNum:rows]
        # print(trainIndex)
        # print(testIndex)
        trainData = data[trainIndex]
        testData = data[testIndex]
        return trainData,testData

def predict(X,Y,neibers=3):
    if len(X)==0:
        print('There is no input!')
        return []
    else:
        # get minimum Ks dis
        # print(np.argpartition(-1*np.array(dis_x1_x2),-K)[:,-K:])
        minIndex = np.argpartition(-1*np.array(dis_x1_x2),-neibers)[:,-neibers:]
        # print(minIndex[0,:],train_y_target[minIndex[0,:]])
        predictTagAll = []
        for i in range(len(X)):
            predictTagAll.append(Y[minIndex[i,:]])
        mostTag,mostCount=mode(np.array(predictTagAll)[:,:],axis=1)
        confidenceDegree = mostCount / neibers
        # print('cond=',confidenceDegree,'\n')
        # print(np.array([mostTag,confidenceDegree]).T.shape)
        return mostTag,confidenceDegree

def evalueateModel():
    print(1)


if __name__ == '__main__':
    default_dir = r"C:\Users\lenovo\Desktop"  # 设置默认打开目录
    # strFile = tkinter.filedialog.askopenfilename(title=u"选择文件",
    #                                  initialdir=(os.path.expanduser(default_dir)))
    strFile=r"C:\Users\win 10\Desktop\hell\datingTestSet2.txt"
    if strFile=="":
        print("Pleace select file path!")
    else:
        data = loadData(strFile)
        for i in range(10):
            trainData,testData = trainTestSplit(data)
            [row,col] = np.array(trainData).shape
            train_x_input = trainData[0:,0:col-1]
            train_y_target = trainData[:,col-1]
            test_x_input = testData[:,0:col-1]
            test_y_target = testData[:,col-1]
            # print('x_test:',x_test)
            dis_x1_x2 = []
            for i in range(len(test_x_input)):
                dis_x1_x2.append(calculate_distance(train_x_input,test_x_input[i,:]))
            K=3
            preTag,preCon = predict(dis_x1_x2,train_y_target,K)
            # print(preTag,preCon)
            # print(test_y_target[0:10])
            preResult = np.reshape(preTag,(len(preTag)))-test_y_target.T
            # print(preResult)
            print(np.sum(preResult==0),np.sum(preResult==0)/len(test_y_target))

            fig = plt.figure()
            plt.figure(22)
            plt.subplot(221) #分成4个格子，占第一个
            plt.scatter(train_x_input[:,0],train_x_input[:,1],15.0*array(train_y_target), 15.0*array(train_y_target))

            plt.subplot(222)
            plt.scatter(train_x_input[:,0],train_x_input[:,2],15.0*array(train_y_target), 15.0*array(train_y_target))

            plt.subplot(212)
            plt.scatter(train_x_input[:,1],train_x_input[:,2],15.0*array(train_y_target), 15.0*array(train_y_target))

            # fig2 = plt.figure()
            # ax2 = fig2.add_subplot(111)
            # ax2.scatter(train_x_input[:,0],train_x_input[:,2],15.0*array(train_y_target), 15.0*array(train_y_target))
            plt.show()
