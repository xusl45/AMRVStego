from xgboost import XGBClassifier
from numpy import loadtxt
from lightgbm import LGBMClassifier
from numpy import sort
import numpy as np
import random
import sklearn
import csv
import heapq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
import joblib
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import os
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
import csv
import json
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#############################################################################
pluseDict ={}
plusePairDict1 = {0:[(0,0)],1:[(0,1),(1,0)],2:[(0,2),(2,0)],3:[(0,3),(3,0)],4:[(0,4),(4,0)],5:[(0,5),(5,0)],6:[(0,6),(6,0)],7:[(0,7),(7,0)]}        #？？？？？
plusePairDict2 = {8:[(1,1)],9:[(1,2),(2,1)],10:[(1,3),(3,1)],11:[(1,4),(4,1)],12:[(1,5),(5,1)],13:[(1,6),(6,1)],14:[(1,7),(7,1)]}
plusePairDict3 = {15:[(2,2)],16:[(2,3),(3,2)],17:[(2,4),(4,2)],18:[(2,5),(5,2)],19:[(2,6),(6,2)],20:[(2,7),(7,2)]}
plusePairDict4 = {21:[(3,3)],22:[(3,4),(4,3)],23:[(3,5),(5,3)],24:[(3,6),(6,3)],25:[(3,7),(7,3)],26:[(4,4)],27:[(4,5),(5,4)]}
plusePairDict5 = {28:[(4,6),(6,4)],29:[(4,7),(7,4)],30:[(5,5)],31:[(5,6),(6,5)],32:[(5,7),(7,5)],33:[(6,6)],34:[(6,7),(7,6)],35:[(7,7)]}
pluseDict.update(plusePairDict1)
pluseDict.update(plusePairDict2)
pluseDict.update(plusePairDict3)
pluseDict.update(plusePairDict4)
pluseDict.update(plusePairDict5)

def test(urlFileIn):#此处是正常提取10s固定码本脉冲的函数，urlFile代表输入的脉冲序列文件
    pluseList1,pluseList2,pluseList3,pluseList4,pluseList5,pluseList6,pluseList7,pluseList8,pluseList9,pluseList10 = [],[],[],[],[],[],[],[],[],[]
    all_feature = []
    allpluse = []
    feature2 = []
    with open(urlFileIn, 'r') as fin:
        for line in fin.readlines():
            sub_pluse = []
            pluses = [int(i) for i in line.strip("\n").split(",")[:10]]
            sub_pluse.append(pluses[0]//5)      #？？？
            sub_pluse.append((pluses[2]-1)//5)
            sub_pluse.append((pluses[4]-2)//5)
            sub_pluse.append((pluses[6]-3)//5)
            sub_pluse.append((pluses[8]-4)//5)
            sub_pluse.append(pluses[1]//5)
            sub_pluse.append((pluses[3]-1)//5)
            sub_pluse.append((pluses[5]-2)//5)
            sub_pluse.append((pluses[7]-3)//5)
            sub_pluse.append((pluses[9]-4)//5)
            pluseList1.append(pluses[0]//5)
            pluseList6.append(pluses[1]//5)
            pluseList2.append((pluses[2]-1)//5)
            pluseList7.append((pluses[3]-1)//5)
            pluseList3.append((pluses[4]-2)//5)
            pluseList8.append((pluses[5]-2)//5)
            pluseList4.append((pluses[6]-3)//5)
            pluseList9.append((pluses[7]-3)//5)
            pluseList5.append((pluses[8]-4)//5)
            pluseList10.append((pluses[9]-4)//5)
            allpluse.append(sub_pluse)
    ##########################################################################################
    feature1 = extract_SPC(allpluse) #计算SPC特征
    ############################################################
    feature2_1 = extract_TPC(pluseList1,pluseList6)
    feature2_2 = extract_TPC(pluseList2,pluseList7)
    feature2_3 = extract_TPC(pluseList3,pluseList8)
    feature2_4 = extract_TPC(pluseList4,pluseList9)
    feature2_5 = extract_TPC(pluseList5,pluseList10)
    ########################################################################
    for i in range(len(feature2_1)):
        ave_data = (feature2_1[i]+feature2_2[i]+feature2_3[i]+feature2_4[i]+feature2_5[i])/5
        feature2.append(ave_data)
    all_feature.extend(feature1)
    all_feature.extend(feature2)
    print(len(all_feature))
    return all_feature

def extract_SPC(allPluseList):
    L = 10
    combination_num = 45  #？？？？？
    feature_num = 36
    all_count_pairs = np.zeros(shape=feature_num)
    for subLen in range(len(allPluseList)):
        coutPairs = np.zeros(shape=feature_num)
        for i in range(L):
            pluse_x = allPluseList[subLen][i]
            for j in range(i+1,L):
                pluse_y = allPluseList[subLen][j]
                for key in pluseDict.keys():   #？？？？
                    pair = (pluse_x,pluse_y)
                    if pair in pluseDict[key]:
                        coutPairs[key] += 1
                        break
        pro_list = []
        for countTime in coutPairs:
            if countTime == 0:
                pro_list.append(0)
            else:
                pro_list.append(math.log2(float(countTime/combination_num)))
        for slen in range(feature_num):
            all_count_pairs[slen] += pro_list[slen]
    final_pro_list = [float(pro/len(allPluseList)) for pro in all_count_pairs]
    #print(final_pro_list)
    #print(len(final_pro_list))
    return final_pro_list

def extract_TPC(pluseList1,pluseList2):
    #print("TPC feature")
    pair_dict = {}
    numPluse = 8   #？？？？
    pairsNum = 64
    single_num_pluse = 8
    num_pluse1 = np.zeros(shape=single_num_pluse) #对应第一个脉冲序列的概率
    num_pluse2 = np.zeros(shape=single_num_pluse) #对应第二个脉冲序列的概率
    for i in range(numPluse):
        for j in range(numPluse):
            pair = (i,j)
            pair_dict[i*8+j]=pair
    #print(pair_dict)
    for i in range(len(pluseList1)):
        num_pluse1[pluseList1[i]] += 1
    #print(num_pluse1)
    pro_pluse1 = [float(num/len(pluseList1)) for num in num_pluse1] #这个对应为每一个轨道中的第一个脉冲值的概率
    for j in range(len(pluseList2)):
        num_pluse2[pluseList2[j]] += 1
    pro_pluse2 = [float(num/len(pluseList2)) for num in num_pluse2] #这个对应为每一个轨道中的第二个脉冲值的概率
    countPairs = np.zeros(shape=pairsNum)   #？？？
    joint_pro = np.zeros(shape=(pairsNum))
    for subLen in range(len(pluseList1)):
        pairs = (pluseList1[subLen],pluseList2[subLen])
        for key in pair_dict.keys():
            if pairs == pair_dict[key]:
                countPairs[key] += 1
                break
    #print(countPairs)
    countPairs = [float(num/len(pluseList1)) for num in countPairs]
    for i in range(numPluse):
        for j in range(numPluse):
            judge_value = float(countPairs[i*8+j]/(pro_pluse1[i]*pro_pluse2[j]))
            if judge_value == 0:
                joint_pro[i*8+j] = 0
            else:
                joint_pro[i*8+j] = countPairs[i*8+j]*math.log2(float(countPairs[i*8+j]/(pro_pluse1[i]*pro_pluse2[j]))) #计算互信息
    #print(joint_pro)
    #print(len(joint_pro))
    return joint_pro #返回一个轨道的互信息


def cal_FA(pre_list, label):
    n_len = len(pre_list)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(n_len):
        if (pre_list[i] == 0 and label[i] == 0):
            TN += 1
        if (pre_list[i] == 0 and label[i] == 1):
            FN += 1
        if (pre_list[i] == 1 and label[i] == 1):
            TP += 1
        if (pre_list[i] == 1 and label[i] == 0):
            FP += 1
    FA = FN/(FN+TN)
    R_call_miss = 1 - TN/(TN+FP)
    return FA, R_call_miss

def normalization(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = np.array(data)
    X_minMax = min_max_scaler.fit_transform(data)
    return X_minMax

def norm(data):
    _range = np.max(data)-np.min(data)
    x = (data-np.min(data))/_range
    print("max min")
    print(np.max(data))
    print(np.min(data))
    return x

def transform_data(data,minValue, maxValue):
    _range = maxValue - minValue
    x = (data-minValue)/_range
    return x

def train_SVM(X_train,y_train,modelPath):
    #model_svm = SklearnClassifier(SVC())
    model_svm = sklearn.svm.SVC(kernel='linear',probability=True)
    model_svm.fit(X_train,y_train)
    joblib.dump(model_svm,modelPath)
    return model_svm

def train_GBDT(X_train,y_train,modelPath):
    model_GBDT = GradientBoostingClassifier(n_estimators=100,max_depth=4)
    model_GBDT.fit(X_train,y_train)
    joblib.dump(model_GBDT,modelPath)
    return model_GBDT

def train_LGBM(X_train,y_train,modelPath):
    model_LGBM = LGBMClassifier(max_depth=4,n_jobs=3,num_leaves=32)
    model_LGBM.fit(X_train,y_train)
    joblib.dump(model_LGBM,modelPath)
    return model_LGBM

def train_RF(X_train,y_train,modelPath):
    model_RF = RandomForestClassifier(n_estimators=100,max_depth=4,n_jobs=3)
    model_RF.fit(X_train,y_train)
    joblib.dump(model_RF,modelPath)
    return model_RF

def train_XGBoost(X_train,y_train,modelPath):
    model_XGBoost = XGBClassifier(n_estimators=100,max_depth=4,n_jobs=3)
    model_XGBoost.fit(X_train,y_train)
    joblib.dump(model_XGBoost,modelPath)#对应训练第一阶段的XGBoost_model_1
    return model_XGBoost

def train_MLP(X_train,y_train,modelPath):
    model_MLP = MLPClassifier(hidden_layer_sizes=25)
    model_MLP.fit(X_train,y_train)
    joblib.dump(model_MLP,modelPath)
    return model_MLP

def train_SVM_third_stage(X_train,y_train,modelPath):
    #model_svm = SklearnClassifier(SVC())
    model_svm = sklearn.svm.SVC(kernel='linear',probability=True)
    model_svm.fit(X_train,y_train)
    joblib.dump(model_svm,modelPath)
    return model_svm

def train_GBDT_third_stage(X_train,y_train,modelPath):
    model_GBDT = GradientBoostingClassifier(n_estimators=50,max_depth=2)
    model_GBDT.fit(X_train,y_train)
    joblib.dump(model_GBDT,modelPath)
    return model_GBDT

def train_LGBM_third_stage(X_train,y_train,modelPath):
    model_LGBM = LGBMClassifier(max_depth=2,n_jobs=3,n_estimators=50)
    model_LGBM.fit(X_train,y_train)
    joblib.dump(model_LGBM,modelPath)
    return model_LGBM

def train_RF_third_stage(X_train,y_train,modelPath):
    model_RF = RandomForestClassifier(n_estimators=50,max_depth=2,n_jobs=3)
    model_RF.fit(X_train,y_train)
    joblib.dump(model_RF,modelPath)
    return model_RF

def train_XGBoost_third_stage(X_train,y_train,modelPath):
    model_XGBoost = XGBClassifier(n_estimators=50,max_depth=2,n_jobs=3)
    model_XGBoost.fit(X_train,y_train)
    joblib.dump(model_XGBoost,modelPath)#对应训练第一阶段的XGBoost_model_1
    return model_XGBoost

def train_MLP_third_stage(X_train,y_train,modelPath):
    model_MLP = MLPClassifier(hidden_layer_sizes=25)
    model_MLP.fit(X_train,y_train)
    joblib.dump(model_MLP,modelPath)
    return model_MLP

def detection_sample_Geiser_and_Miao2_and_Miao4(modelDirFile, feature_data):#这里对应单个样本检测入口，针对Geiser,Miao-2,Miao-4的隐蔽通信方法检测，且数据进过归一化处理，feature_data表示提取的样本特征
    result= []
    num_classifier = 6
    cross_num = 5
    maxValue = 0.14850828190110477
    minValue = -2.793001211549345
    feature_data = np.array(transform_data(feature_data,minValue,maxValue)).reshape(1,-1)#对数据进行归一化处理
    #########################################################################
    print("first stage")
    for i in range(num_classifier):
        if i == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_GBDT','.pickle')
            GBDT_model = joblib.load(urlFile)
            GBDT_1_result = GBDT_model.predict(feature_data)
            result.append(GBDT_1_result)
        if i == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_XGBoost','.pickle')
            XGBoost_1_model = joblib.load(urlFile)
            XGBoost_1_result = XGBoost_1_model.predict(feature_data)
            result.append(XGBoost_1_result)
        if i == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_LGBM','.pickle')
            LGBM_model = joblib.load(urlFile)
            LGBM_1_result = LGBM_model.predict(feature_data)
            result.append(LGBM_1_result)
        if i == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_RF','.pickle')
            RF_model = joblib.load(urlFile)
            RF_1_result = RF_model.predict(feature_data)
            result.append(RF_1_result)
        if i == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_SVM','.pickle')
            SVM_model = joblib.load(urlFile)
            SVM_1_result = SVM_model.predict(feature_data)
            result.append(SVM_1_result)
        if i == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_MLP','.pickle')
            MLP_model = joblib.load(urlFile)
            MLP_result = MLP_model.predict(feature_data)
            result.append(MLP_result)
    ##################################################################################################
    print("second stage")
    third_feature = []
    flag = 0
    for i in range(cross_num):
        for j in range(num_classifier):
            if (j == 0):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_GBDT','.pickle')
                GBDT_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_GBDT = GBDT_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_GBDT)
                GBDT_2_result = GBDT_2_model.predict(feature_data)#这里得到对应的分类结果,
                result.append(GBDT_2_result)
            if (j == 1):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_XGBoost','.pickle')
                XGBoost_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_XGBoost = XGBoost_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_XGBoost)
                XGBoost_2_result = XGBoost_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(XGBoost_2_result)
            if (j == 2):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_LGBM','.pickle')
                LGBM_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_LGBM = LGBM_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_LGBM)
                LGBM_2_result = LGBM_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(LGBM_2_result)
            if (j == 3):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_RF','.pickle')
                RF_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_RF = RF_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_RF)
                RF_2_result = RF_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(RF_2_result)
            if (j == 4):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_SVM','.pickle')
                SVM_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_SVM = SVM_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_SVM)
                SVM_2_result = SVM_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(SVM_2_result)
            if (j == 5):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_MLP','.pickle')
                MLP_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_MLP = MLP_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_MLP)
                MLP_2_result = MLP_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(MLP_2_result)
        flag = 1
        third_feature = np.array(third_feature).transpose()
        third_feature = third_feature.tolist()#这里得到对应的最终的6维特征
        #######################################################################################
    print("third stage")
    for third_j in range(num_classifier):
        if third_j == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_GBDT','.pickle')
            GBDT_3_model = joblib.load(urlFile)
            GBDT_3_result = GBDT_3_model.predict(third_feature)
            result.append(GBDT_3_result)
        if third_j == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_XGBoost','.pickle')
            XGBoost_3_model = joblib.load(urlFile)
            XGBoost_3_result = XGBoost_3_model.predict(third_feature)
            result.append(XGBoost_3_result)
        if third_j == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_LGBM','.pickle')
            LGBM_3_model = joblib.load(urlFile)
            LGBM_3_result = LGBM_3_model.predict(third_feature)
            result.append(LGBM_3_result)
        if third_j == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_RF','.pickle')
            RF_3_model = joblib.load(urlFile)
            RF_3_result = RF_3_model.predict(third_feature)
            result.append(RF_3_result)
        if third_j == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_SVM','.pickle')
            SVM_3_model = joblib.load(urlFile)
            SVM_3_result = SVM_3_model.predict(third_feature)
            result.append(SVM_3_result)
        if third_j == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_MLP','.pickle')
            MLP_3_model = joblib.load(urlFile)
            MLP_3_result = MLP_3_model.predict(third_feature)
            result.append(MLP_3_result)
    print("second stage finished")
    print("third stage data")
    result = np.array(result).transpose().astype(int)
    single_final_result = np.argmax(np.bincount(result[0]))
    print("single final result")
    print(single_final_result)
    ####################################################################################

def detection_sample_Miao1(modelDirFile, feature_data):#这里对应单个样本检测入口，针对Miao-1的隐蔽通信方法检测，且数据进行了归一化处理，feature_data表示提取的样本特征
    print(modelDirFile)
    result= []
    num_classifier = 6
    cross_num = 5
    maxValue = 0.15075279001807002
    minValue = -2.793001211549345
    feature_data = np.array(transform_data(feature_data,minValue,maxValue)).reshape(1,-1)#对数据进行归一化处理
    #########################################################################
    print("first stage")
    print(modelDirFile)
    for i in range(num_classifier):
        if i == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_GBDT','.pickle')
            GBDT_model = joblib.load(urlFile)
            GBDT_1_result = GBDT_model.predict(feature_data)
            result.append(GBDT_1_result)
        if i == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_XGBoost','.pickle')
            XGBoost_1_model = joblib.load(urlFile)
            XGBoost_1_result = XGBoost_1_model.predict(feature_data)
            result.append(XGBoost_1_result)
        if i == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_LGBM','.pickle')
            LGBM_model = joblib.load(urlFile)
            LGBM_1_result = LGBM_model.predict(feature_data)
            result.append(LGBM_1_result)
        if i == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_RF','.pickle')
            RF_model = joblib.load(urlFile)
            RF_1_result = RF_model.predict(feature_data)
            result.append(RF_1_result)
        if i == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_SVM','.pickle')
            SVM_model = joblib.load(urlFile)
            SVM_1_result = SVM_model.predict(feature_data)
            result.append(SVM_1_result)
        if i == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_MLP','.pickle')
            MLP_model = joblib.load(urlFile)
            MLP_result = MLP_model.predict(feature_data)
            result.append(MLP_result)
    ##################################################################################################
    print("second stage")
    third_feature = []
    flag = 0
    for i in range(cross_num):
        for j in range(num_classifier):
            if (j == 0):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_GBDT','.pickle')
                GBDT_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_GBDT = GBDT_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_GBDT)
                GBDT_2_result = GBDT_2_model.predict(feature_data)#这里得到对应的分类结果,
                result.append(GBDT_2_result)
            if (j == 1):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_XGBoost','.pickle')
                XGBoost_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_XGBoost = XGBoost_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_XGBoost)
                XGBoost_2_result = XGBoost_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(XGBoost_2_result)
            if (j == 2):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_LGBM','.pickle')
                LGBM_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_LGBM = LGBM_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_LGBM)
                LGBM_2_result = LGBM_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(LGBM_2_result)
            if (j == 3):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_RF','.pickle')
                RF_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_RF = RF_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_RF)
                RF_2_result = RF_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(RF_2_result)
            if (j == 4):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_SVM','.pickle')
                SVM_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_SVM = SVM_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_SVM)
                SVM_2_result = SVM_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(SVM_2_result)
            if (j == 5):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_MLP','.pickle')
                MLP_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_MLP = MLP_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_MLP)
                MLP_2_result = MLP_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(MLP_2_result)
        flag = 1
        third_feature = np.array(third_feature).transpose()
        third_feature = third_feature.tolist()#这里得到对应的最终的6维特征
        #######################################################################################
    print("third stage")
    for third_j in range(num_classifier):
        if third_j == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_GBDT','.pickle')
            GBDT_3_model = joblib.load(urlFile)
            GBDT_3_result = GBDT_3_model.predict(third_feature)
            result.append(GBDT_3_result)
        if third_j == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_XGBoost','.pickle')
            XGBoost_3_model = joblib.load(urlFile)
            XGBoost_3_result = XGBoost_3_model.predict(third_feature)
            result.append(XGBoost_3_result)
        if third_j == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_LGBM','.pickle')
            LGBM_3_model = joblib.load(urlFile)
            LGBM_3_result = LGBM_3_model.predict(third_feature)
            result.append(LGBM_3_result)
        if third_j == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_RF','.pickle')
            RF_3_model = joblib.load(urlFile)
            RF_3_result = RF_3_model.predict(third_feature)
            result.append(RF_3_result)
        if third_j == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_SVM','.pickle')
            SVM_3_model = joblib.load(urlFile)
            SVM_3_result = SVM_3_model.predict(third_feature)
            result.append(SVM_3_result)
        if third_j == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_MLP','.pickle')
            MLP_3_model = joblib.load(urlFile)
            MLP_3_result = MLP_3_model.predict(third_feature)
            result.append(MLP_3_result)
    print("second stage finished")
    print("third stage data")
    result = np.array(result).transpose().astype(int)
    single_final_result = np.argmax(np.bincount(result[0]))
    print("single final result")
    print(single_final_result)
    ####################################################################################

def detection_sample_without_normalzation(modelDirFile, feature_data):#这里对应单个样本测试，数据都没有经过归一化处理
    result= []
    num_classifier = 6
    cross_num = 5
    feature_data = np.array(feature_data).reshape(1,-1)
    #########################################################################
    print("first stage")
    for i in range(num_classifier):
        if i == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_GBDT','.pickle')
            GBDT_model = joblib.load(urlFile)
            GBDT_1_result = GBDT_model.predict(feature_data)
            result.append(GBDT_1_result)
        if i == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_XGBoost','.pickle')
            XGBoost_1_model = joblib.load(urlFile)
            XGBoost_1_result = XGBoost_1_model.predict(feature_data)
            result.append(XGBoost_1_result)
        if i == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_LGBM','.pickle')
            LGBM_model = joblib.load(urlFile)
            LGBM_1_result = LGBM_model.predict(feature_data)
            result.append(LGBM_1_result)
        if i == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_RF','.pickle')
            RF_model = joblib.load(urlFile)
            RF_1_result = RF_model.predict(feature_data)
            result.append(RF_1_result)
        if i == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_SVM','.pickle')
            SVM_model = joblib.load(urlFile)
            SVM_1_result = SVM_model.predict(feature_data)
            result.append(SVM_1_result)
        if i == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_MLP','.pickle')
            MLP_model = joblib.load(urlFile)
            MLP_result = MLP_model.predict(feature_data)
            result.append(MLP_result)
    ##################################################################################################
    print("second stage")
    third_feature = []
    flag = 0
    for i in range(cross_num):
        for j in range(num_classifier):
            if (j == 0):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_GBDT','.pickle')
                GBDT_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_GBDT = GBDT_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_GBDT)
                GBDT_2_result = GBDT_2_model.predict(feature_data)#这里得到对应的分类结果,
                result.append(GBDT_2_result)
            if (j == 1):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_XGBoost','.pickle')
                XGBoost_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_XGBoost = XGBoost_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_XGBoost)
                XGBoost_2_result = XGBoost_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(XGBoost_2_result)
            if (j == 2):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_LGBM','.pickle')
                LGBM_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_LGBM = LGBM_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_LGBM)
                LGBM_2_result = LGBM_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(LGBM_2_result)
            if (j == 3):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_RF','.pickle')
                RF_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_RF = RF_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_RF)
                RF_2_result = RF_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(RF_2_result)
            if (j == 4):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_SVM','.pickle')
                SVM_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_SVM = SVM_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_SVM)
                SVM_2_result = SVM_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(SVM_2_result)
            if (j == 5):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_MLP','.pickle')
                MLP_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_MLP = MLP_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_MLP)
                MLP_2_result = MLP_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(MLP_2_result)
        flag = 1
        third_feature = np.array(third_feature).transpose()
        third_feature = third_feature.tolist()#这里得到对应的最终的6维特征
        #######################################################################################
    print("third stage")
    for third_j in range(num_classifier):
        if third_j == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_GBDT','.pickle')
            GBDT_3_model = joblib.load(urlFile)
            GBDT_3_result = GBDT_3_model.predict(third_feature)
            result.append(GBDT_3_result)
        if third_j == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_XGBoost','.pickle')
            XGBoost_3_model = joblib.load(urlFile)
            XGBoost_3_result = XGBoost_3_model.predict(third_feature)
            result.append(XGBoost_3_result)
        if third_j == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_LGBM','.pickle')
            LGBM_3_model = joblib.load(urlFile)
            LGBM_3_result = LGBM_3_model.predict(third_feature)
            result.append(LGBM_3_result)
        if third_j == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_RF','.pickle')
            RF_3_model = joblib.load(urlFile)
            RF_3_result = RF_3_model.predict(third_feature)
            result.append(RF_3_result)
        if third_j == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_SVM','.pickle')
            SVM_3_model = joblib.load(urlFile)
            SVM_3_result = SVM_3_model.predict(third_feature)
            result.append(SVM_3_result)
        if third_j == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_MLP','.pickle')
            MLP_3_model = joblib.load(urlFile)
            MLP_3_result = MLP_3_model.predict(third_feature)
            result.append(MLP_3_result)
    print("second stage finished")
    print("third stage data")
    result = np.array(result).transpose().astype(int)
    print(result.shape)
    print(result)
    single_final_result = np.argmax(np.bincount(result[0]))
    print("single final result")
    print(single_final_result)
    ####################################################################################

def batch_sample_test(modelDirFile,feature_data, label_data):#样本批量测试，样本没有经过归一化处理
    result= []
    num_classifier = 6
    cross_num = 5
    #########################################################################
    print("first stage")
    for i in range(num_classifier):
        if i == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_GBDT','.pickle')
            GBDT_model = joblib.load(urlFile)
            GBDT_1_result = GBDT_model.predict(feature_data)
            result.append(GBDT_1_result)
        if i == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_XGBoost','.pickle')
            XGBoost_1_model = joblib.load(urlFile)
            XGBoost_1_result = XGBoost_1_model.predict(feature_data)
            result.append(XGBoost_1_result)
        if i == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_LGBM','.pickle')
            LGBM_model = joblib.load(urlFile)
            LGBM_1_result = LGBM_model.predict(feature_data)
            result.append(LGBM_1_result)
        if i == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_RF','.pickle')
            RF_model = joblib.load(urlFile)
            RF_1_result = RF_model.predict(feature_data)
            result.append(RF_1_result)
        if i == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_SVM','.pickle')
            SVM_model = joblib.load(urlFile)
            SVM_1_result = SVM_model.predict(feature_data)
            result.append(SVM_1_result)
        if i == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_MLP','.pickle')
            MLP_model = joblib.load(urlFile)
            MLP_result = MLP_model.predict(feature_data)
            result.append(MLP_result)
    ##################################################################################################
    print("second stage")
    third_feature = []
    flag = 0
    for i in range(cross_num):
        for j in range(num_classifier):
            if (j == 0):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_GBDT','.pickle')
                GBDT_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_GBDT = GBDT_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_GBDT)
                GBDT_2_result = GBDT_2_model.predict(feature_data)#这里得到对应的分类结果,
                result.append(GBDT_2_result)
            if (j == 1):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_XGBoost','.pickle')
                XGBoost_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_XGBoost = XGBoost_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_XGBoost)
                XGBoost_2_result = XGBoost_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(XGBoost_2_result)
            if (j == 2):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_LGBM','.pickle')
                LGBM_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_LGBM = LGBM_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_LGBM)
                LGBM_2_result = LGBM_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(LGBM_2_result)
            if (j == 3):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_RF','.pickle')
                RF_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_RF = RF_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_RF)
                RF_2_result = RF_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(RF_2_result)
            if (j == 4):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_SVM','.pickle')
                SVM_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_SVM = SVM_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_SVM)
                SVM_2_result = SVM_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(SVM_2_result)
            if (j == 5):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_MLP','.pickle')
                MLP_2_model = joblib.load(urlFile)
                if (flag == 0):
                    pre_MLP = MLP_2_model.predict_proba(feature_data)[:,0]
                    third_feature.append(pre_MLP)
                MLP_2_result = MLP_2_model.predict(feature_data)#这里得到对应的第二个分类器集合的预测结果
                result.append(MLP_2_result)
        flag = 1
        third_feature = np.array(third_feature).transpose()
        third_feature = third_feature.tolist()#这里得到对应的最终的6维特征
        #######################################################################################
    print("third stage")
    for third_j in range(num_classifier):
        if third_j == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_GBDT','.pickle')
            GBDT_3_model = joblib.load(urlFile)
            GBDT_3_result = GBDT_3_model.predict(third_feature)
            result.append(GBDT_3_result)
        if third_j == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_XGBoost','.pickle')
            XGBoost_3_model = joblib.load(urlFile)
            XGBoost_3_result = XGBoost_3_model.predict(third_feature)
            result.append(XGBoost_3_result)
        if third_j == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_LGBM','.pickle')
            LGBM_3_model = joblib.load(urlFile)
            LGBM_3_result = LGBM_3_model.predict(third_feature)
            result.append(LGBM_3_result)
        if third_j == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_RF','.pickle')
            RF_3_model = joblib.load(urlFile)
            RF_3_result = RF_3_model.predict(third_feature)
            result.append(RF_3_result)
        if third_j == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_SVM','.pickle')
            SVM_3_model = joblib.load(urlFile)
            SVM_3_result = SVM_3_model.predict(third_feature)
            result.append(SVM_3_result)
        if third_j == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_MLP','.pickle')
            MLP_3_model = joblib.load(urlFile)
            MLP_3_result = MLP_3_model.predict(third_feature)
            result.append(MLP_3_result)
    print("second stage finished")
    print("third stage data")
    result = np.array(result).transpose().astype(int)
    print(result.shape)
    print(result)
    final_pre_res = []
    for line in result:
        final_pre_res.append(int(np.argmax(np.bincount(line))))
    accuracy = accuracy_score(label_data,final_pre_res)
    print("test accuracy")
    print(accuracy)
    FA,FR = cal_FA(final_pre_res,label_data)
    print("虚警率，漏检率")
    print(FA)
    print(FR)

def multi_combination_classifer_train_dataNorm(modelDirFile, featurePath):#数据经过归一化处理后的调用方法

    cross_num = 5
    num_classifier = 6
    dataset1 = loadtxt(featurePath,delimiter=",")
    x_data = dataset1[1:,0:100]
    x_nor_data = norm(x_data)
    print("junzhi")
    print(x_data.mean())
    print(x_data.std())
    y_data = dataset1[1:,100]
    #print(x_data)
    #print(y_data)
    ############################################################################################################
    random_state = random.randint(1,100000)
    X_train, X_test, y_train, y_test = train_test_split(x_nor_data,y_data,test_size=0.25,random_state=random_state)
    print("load data successfully")
    ##############################################################################################################
    print("first stage")
    for i in range(num_classifier):
        if i == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_GBDT','.pickle')
            GBDT_model = train_GBDT(X_train,y_train,urlFile)
        if i == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_XGBoost','.pickle')
            XGBoost_model = train_XGBoost(X_train,y_train,urlFile)
        if i == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_LGBM','.pickle')
            LGBM_model = train_LGBM(X_train,y_train,urlFile)
        if i == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_RF','.pickle')
            RF_model = train_RF(X_train,y_train,urlFile)
        if i == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_SVM','.pickle')
            SVM_model = train_SVM(X_train,y_train,urlFile)
        if i == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_MLP','.pickle')
            MLP_model = train_MLP(X_train,y_train,urlFile)
    print("finished")
    ################################################################################################################
    print("second stage")
    third_feature = []
    third_stage_label = []
    for i in range(cross_num):
        random_state = random.randint(1,100000)
        X_train2,X_test2, y_train2, y_test2 = train_test_split(X_train,y_train,test_size=0.2,random_state=random_state)
        #print(X_train2)
        #print(y_train2)
        each_third_feature = []
        for j in range(num_classifier):
            if (j == 0):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_GBDT','.pickle')
                GBDT_2_model = train_GBDT(X_train2,y_train2,urlFile)
                pre_GBDT = GBDT_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果,1280*1,
                #print(pre_GBDT)
                each_third_feature.append(pre_GBDT)
            if (j == 1):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_XGBoost','.pickle')
                XGBoost_2_model = train_XGBoost(X_train2,y_train2,urlFile)
                pre_XGBoost = XGBoost_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_XGBoost)
            if (j == 2):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_LGBM','.pickle')
                LGBM_2_model = train_LGBM(X_train2,y_train2,urlFile)
                pre_LGBM = LGBM_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_LGBM)
            if (j == 3):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_RF','.pickle')
                RF_2_model = train_RF(X_train2,y_train2,urlFile)
                pre_RF = RF_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_RF)
            if (j == 4):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_SVM','.pickle')
                SVM_2_model = train_SVM_third_stage(X_train2,y_train2,urlFile)
                pre_SVM = SVM_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_SVM)
            if (j == 5):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_MLP','.pickle')
                MLP_2_model = train_MLP(X_train2,y_train2,urlFile)
                pre_MLP = MLP_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_MLP)
        each_third_feature = np.array(each_third_feature).transpose()
        each_third_feature = each_third_feature.tolist()
        third_feature.append(each_third_feature)
        third_stage_label.extend(y_test2)
    third_feature = np.array(third_feature)
    third_feature = np.reshape(third_feature,(-1,6))
    third_stage_label = np.array(third_stage_label)
    print("second stage finished")
    print("third stage data")
    print(third_feature.shape)
    print(third_stage_label.shape)
    ###############################################################################################################
    for i in range(num_classifier):
        if i == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_GBDT','.pickle')
            GBDT_model = train_GBDT_third_stage(third_feature,third_stage_label,urlFile)
        if i == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_XGBoost','.pickle')
            XGBoost_model = train_XGBoost_third_stage(third_feature,third_stage_label,urlFile)
        if i == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_LGBM','.pickle')
            LGBM_model = train_LGBM_third_stage(third_feature,third_stage_label,urlFile)
        if i == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_RF','.pickle')
            RF_model = train_RF_third_stage(third_feature,third_stage_label,urlFile)
        if i == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_SVM','.pickle')
            SVM_model = train_SVM_third_stage(third_feature,third_stage_label,urlFile)
        if i == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_MLP','.pickle')
            MLP_model = train_MLP_third_stage(third_feature,third_stage_label,urlFile)
    print("finished")
    print("third stage finished")
    batch_sample_test(modelDirFile,X_test,y_test)



def multi_combination_classifer_train(modelDirFile, featurePath):#这里数据没有经过归一化处理，直接进行模型训练和随机批次的样本训练

    cross_num = 5
    num_classifier = 6
    print(featurePath)
    dataset1 = loadtxt(featurePath,delimiter=",")
    x_data = dataset1[1:,0:100]
    #x_nor_data = norm(x_data)
    print("junzhi")
    print(x_data.mean())
    print(x_data.std())
    y_data = dataset1[1:,100]
    #print(x_data)
    #print(y_data)
    ############################################################################################################
    random_state = random.randint(1,100000)
    X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.25,random_state=random_state)
    print("load data successfully")
    ##############################################################################################################
    print("first stage")
    for i in range(num_classifier):
        if i == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_GBDT','.pickle')
            print(urlFile)
            print(len(X_train))
            print(len(y_train))
            GBDT_model = train_GBDT(X_train,y_train,urlFile)
        if i == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_XGBoost','.pickle')
            print(urlFile)
            XGBoost_model = train_XGBoost(X_train,y_train,urlFile)
        if i == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_LGBM','.pickle')
            print(urlFile)

            LGBM_model = train_LGBM(X_train,y_train,urlFile)
        if i == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_RF','.pickle')
            print(urlFile)
            RF_model = train_RF(X_train,y_train,urlFile)
        if i == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_SVM','.pickle')
            print(urlFile)
            SVM_model = train_SVM(X_train,y_train,urlFile)
        if i == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_1','_MLP','.pickle')
            print(urlFile)
            MLP_model = train_MLP(X_train,y_train,urlFile)
    print("finished")
    ################################################################################################################
    print("second stage")
    third_feature = []
    third_stage_label = []
    for i in range(cross_num):
        random_state = random.randint(1,100000)
        X_train2,X_test2, y_train2, y_test2 = train_test_split(X_train,y_train,test_size=0.2,random_state=random_state)
        #print(X_train2)
        #print(y_train2)
        each_third_feature = []
        for j in range(num_classifier):
            if (j == 0):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_GBDT','.pickle')
                GBDT_2_model = train_GBDT(X_train2,y_train2,urlFile)
                pre_GBDT = GBDT_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果,1280*1,
                #print(pre_GBDT)
                each_third_feature.append(pre_GBDT)
            if (j == 1):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_XGBoost','.pickle')
                XGBoost_2_model = train_XGBoost(X_train2,y_train2,urlFile)
                pre_XGBoost = XGBoost_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_XGBoost)
            if (j == 2):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_LGBM','.pickle')
                LGBM_2_model = train_LGBM(X_train2,y_train2,urlFile)
                pre_LGBM = LGBM_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_LGBM)
            if (j == 3):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_RF','.pickle')
                RF_2_model = train_RF(X_train2,y_train2,urlFile)
                pre_RF = RF_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_RF)
            if (j == 4):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_SVM','.pickle')
                SVM_2_model = train_SVM_third_stage(X_train2,y_train2,urlFile)
                pre_SVM = SVM_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_SVM)
            if (j == 5):
                urlFile =r'{0}\{1}{2}{3}{4}{5}{6}'.format(modelDirFile,'model_2_',str(j),'_',str(i), '_MLP','.pickle')
                MLP_2_model = train_MLP(X_train2,y_train2,urlFile)
                pre_MLP = MLP_2_model.predict_proba(X_test2)[:,0]#这里得到对应的第二个分类器集合的预测结果
                each_third_feature.append(pre_MLP)
        each_third_feature = np.array(each_third_feature).transpose()
        each_third_feature = each_third_feature.tolist()
        third_feature.append(each_third_feature)
        third_stage_label.extend(y_test2)
    third_feature = np.array(third_feature)
    third_feature = np.reshape(third_feature,(-1,6))
    third_stage_label = np.array(third_stage_label)
    print("second stage finished")
    print("third stage data")
    print(third_feature.shape)
    print(third_stage_label.shape)
    ###############################################################################################################
    for i in range(num_classifier):
        if i == 0:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_GBDT','.pickle')
            GBDT_model = train_GBDT_third_stage(third_feature,third_stage_label,urlFile)
        if i == 1:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_XGBoost','.pickle')
            XGBoost_model = train_XGBoost_third_stage(third_feature,third_stage_label,urlFile)
        if i == 2:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_LGBM','.pickle')
            LGBM_model = train_LGBM_third_stage(third_feature,third_stage_label,urlFile)
        if i == 3:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_RF','.pickle')
            RF_model = train_RF_third_stage(third_feature,third_stage_label,urlFile)
        if i == 4:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_SVM','.pickle')
            SVM_model = train_SVM_third_stage(third_feature,third_stage_label,urlFile)
        if i == 5:
            urlFile =r'{0}\{1}{2}{3}'.format(modelDirFile,'model_3','_MLP','.pickle')
            MLP_model = train_MLP_third_stage(third_feature,third_stage_label,urlFile)
    print("finished")
    print("third stage finished")
    batch_sample_test(modelDirFile,X_test,y_test)



if __name__=="__main__":
    modelDirFile = r'E:\data\teacherData\model_10_filesave-1\Miao-1'#modelDirFile表示模型表村的目录
    featurePath = r'E:\data\teacherData\Miao-1\Liu_10_train.csv'#传入提取好的特征文件
    # print(modelDirFile)
    multi_combination_classifer_train(modelDirFile,featurePath)#此处训练多组合分类器,数据没有经过归一化处理，
    print(modelDirFile)
    test_sample_path = r'E:\data\ExperimentData\test_data\Miao-1\TXT\rate10\chinese_18.txt'#此处传入了要检测的文件
    single_data_feature = np.array(test(test_sample_path))#single_data_feature表示提取的样本特征
    print(modelDirFile)
    result = detection_sample_Miao1(modelDirFile,single_data_feature)#表示单个样本检测的结果，数据没有经过归一化，检测结果为0时，是嵌入秘密信息的样本，检测为1时，是原始样本，这个与前端展示的不同







