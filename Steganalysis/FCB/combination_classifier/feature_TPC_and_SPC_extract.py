import csv
import json
import math

import numpy as np
######################################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#############################################################################
pluseDict ={}
plusePairDict1 = {0:[(0,0)],1:[(0,1),(1,0)],2:[(0,2),(2,0)],3:[(0,3),(3,0)],4:[(0,4),(4,0)],5:[(0,5),(5,0)],6:[(0,6),(6,0)],7:[(0,7),(7,0)]}
plusePairDict2 = {8:[(1,1)],9:[(1,2),(2,1)],10:[(1,3),(3,1)],11:[(1,4),(4,1)],12:[(1,5),(5,1)],13:[(1,6),(6,1)],14:[(1,7),(7,1)]}
plusePairDict3 = {15:[(2,2)],16:[(2,3),(3,2)],17:[(2,4),(4,2)],18:[(2,5),(5,2)],19:[(2,6),(6,2)],20:[(2,7),(7,2)]}
plusePairDict4 = {21:[(3,3)],22:[(3,4),(4,3)],23:[(3,5),(5,3)],24:[(3,6),(6,3)],25:[(3,7),(7,3)],26:[(4,4)],27:[(4,5),(5,4)]}
plusePairDict5 = {28:[(4,6),(6,4)],29:[(4,7),(7,4)],30:[(5,5)],31:[(5,6),(6,5)],32:[(5,7),(7,5)],33:[(6,6)],34:[(6,7),(7,6)],35:[(7,7)]}
pluseDict.update(plusePairDict1)
pluseDict.update(plusePairDict2)
pluseDict.update(plusePairDict3)
pluseDict.update(plusePairDict4)
pluseDict.update(plusePairDict5)


def extract_feature(urlFileIn):#此处是正常提取10s固定码本脉冲的函数，urlFile代表输入的脉冲序列文件
    pluseList1,pluseList2,pluseList3,pluseList4,pluseList5,pluseList6,pluseList7,pluseList8,pluseList9,pluseList10 = [],[],[],[],[],[],[],[],[],[]
    all_feature = []
    allpluse = []
    feature2 = []
    with open(urlFileIn, 'r') as fin:
        for line in fin.readlines():
            sub_pluse = []
            pluses = [int(i) for i in line.strip("\n").split(",")[:10]]
            sub_pluse.append(pluses[0]//5)
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
    combination_num = 45
    feature_num = 36
    all_count_pairs = np.zeros(shape=feature_num)
    for subLen in range(len(allPluseList)):
        coutPairs = np.zeros(shape=feature_num)
        for i in range(L):
            pluse_x = allPluseList[subLen][i]
            for j in range(i+1,L):
                pluse_y = allPluseList[subLen][j]
                for key in pluseDict.keys():
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
    numPluse = 8
    pairsNum = 64
    single_num_pluse = 8
    num_pluse1 = np.zeros(shape=single_num_pluse) #对应第一个脉冲序列的概率
    num_pluse2 = np.zeros(shape=single_num_pluse) # 对应第二个脉冲序列的概率
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
    countPairs = np.zeros(shape=pairsNum)
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
                joint_pro[i*8+j] = countPairs[i*8+j]*math.log2(float(countPairs[i*8+j]/(pro_pluse1[i]*pro_pluse2[j])))#计算互信息
    #print(joint_pro)
    #print(len(joint_pro))
    return joint_pro #返回一个轨道的互信息

def extract_feature_shortTime(urlFileIn,dataLength):#此处固定码本脉冲的函数，urlFile代表输入的脉冲序列文件，如果要提取不同秒数的特征，就可以控制dataLength(10s对应的dataLength为2000)
    pluseList1,pluseList2,pluseList3,pluseList4,pluseList5,pluseList6,pluseList7,pluseList8,pluseList9,pluseList10 = [],[],[],[],[],[],[],[],[],[]
    all_feature = []
    allpluse = []
    feature2 = []
    n_count = 0
    with open(urlFileIn, 'r') as fin:
        for line in fin.readlines():
            sub_pluse = []
            if n_count < dataLength:
                pluses = [int(i) for i in line.strip("\n").split(",")[:10]]
                sub_pluse.append(pluses[0]//5)
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
            else:
                break
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

def parseFile(dirFileIn,dirFileOut, rate,category = 2,numLen = 12500):
    #此处是提取文件夹下的脉冲序列的总体函数，dirFileIn代表整个原本和隐写样本的上一级文件夹，rate表是嵌入率，category代表四种语音文件目录的索引，numLen一个语音种类下面的文件数目
    count = 0
    with open(r'E:\data\teacherData\Miao-1\Liu_10_train.csv','w',newline='')as csvFile:#此处是将提取存放到哪个.csv文件中
        writer = csv.writer(csvFile)
        begin_row_flag = 1
        fileList = ['Miao-1\TXT\\','Original\TXT\\']
        afile = ["chinese", "English"]
        for eachFile in fileList:
            for i in range(category):
                for j in range(numLen):
                    if (count < 1):
                        urlFileIn =r'{0}\{1}{2}{3}{4}{5}{6}'.format(dirFileIn,eachFile,'rate10\\',afile[i], '_',str(j+1),'.txt')#文件输入路径
                        write_feature = extract_feature(urlFileIn)#写入csv格式时，这里有个坑，要先写好第一行的数据，才能再写入我们要的数据
                        write_feature.append(0)
                        if (begin_row_flag == 1):
                            begin_row = [i for i in range(len(write_feature))]
                            writer.writerow(begin_row)
                            begin_row_flag = 0
                        writer.writerow(write_feature)
                    else:
                        urlFileIn = r'{0}\{1}{2}{3}{4}{5}'.format(dirFileIn,eachFile,afile[i],'_',str(j+1),'.txt')#文件输入路径
                        write_feature = extract_feature(urlFileIn)
                        write_feature.append(1)
                        writer.writerow(write_feature)
            count += 1
            print("finished")

if __name__ == "__main__":
    dirFileIn = r'E:\data\ExperimentData\test_data'
    dirFileOut = 'E:\data\embedingRate'
    parseFile(dirFileIn,dirFileOut,20)