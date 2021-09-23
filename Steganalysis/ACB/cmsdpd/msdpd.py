# coding=utf-8
import numpy as np
import os
import random
import pickle


def read_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    content = []
    for line in lines:
        line = line.replace("\t", " ")
        line_split = line.strip("\r\n\t").strip().split(" ")
        content.append(line_split)
    return content


def get_file_list_lag(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list




def savepkl(path, data):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


def msdpd(lag, msdpd, frames):
    so = np.zeros(frames)
    ms = np.zeros(13)
    zy = np.zeros((13, 13))
    for i in range(frames - 2):
        so[i] = lag[i + 2] - 2 * lag[i + 1] + lag[i]
    for i in range(frames - 2):
        for j in range(-6, 7, 1):
            if so[i] == j:
                ms[j + 6] = ms[j + 6] + 1
    for i in range(frames - 2):
        for l in range(-6, 7, 1):
            for k in range(-6, 7, 1):
                if so[i] == l:
                    if so[i + 1] == k:
                        zy[l + 6][k + 6] = zy[l + 6][k + 6] + 1
    k = 0
    for i in range(13):
        for j in range(13):
            if ms[i] == 0:
                msdpd[k] = 0
            else:
                msdpd[k] = float(zy[i][j]) / ms[i]
            k = k + 1


if __name__ == '__main__':
    stegos = ["yan2","huang","liu"]
    # times = [ "0.8"]
    times = ["1"]
    for stego in stegos:
        for time in times:
            for emr in range(0, 101, 10):
                frames = int(float(time) * 200)
                MSDPD = np.zeros((25000, 170))  # label(1) + msdpd(169)
                lag_path_chinese = "D:\\Vstego\\ACB\\renint\\%s\\Chinese\\%d" % (stego,emr)  # 存放基因延迟参数的路径
                lag_path_english = "D:\\Vstego\\ACB\\renint\\%s\\English\\%d" % (stego, emr)  # 存放基因延迟参数的路径
                lag_path_chinese_test = "D:\\Vstego\\testACB\\renint\\%s\\Chinese\\%d" % (stego, emr)  # 存放基因延迟参数的路径
                lag_path_english_test = "D:\\Vstego\\testACB\\renint\\%s\\English\\%d" % (stego, emr)  # 存放基因延迟参数的路径
                fea_file = "D:\\Vstego\\ACB\\npy\\msdpd-r_%ss_%d_%s.npy" % ( time, emr, stego)
                print("load data from ", lag_path_chinese,lag_path_chinese_test,lag_path_english,lag_path_english_test, "\tsave at ", fea_file)
                if emr == 0:
                    label = 1
                else:
                    label = -1
                lag_file_chinese = get_file_list_lag(lag_path_chinese)  # train
                lag_file_English = get_file_list_lag(lag_path_english)  # train
                lag_file_chinese_test = get_file_list_lag(lag_path_chinese_test)  # test
                lag_file_English_test = get_file_list_lag(lag_path_english_test)  # test
                print(len(lag_file_English))
                print(len(lag_file_English))
                print(len(lag_file_chinese_test))
                print(len(lag_file_English_test))
                all_file = lag_file_English+lag_file_chinese+lag_file_English_test+lag_file_chinese_test
                print(all_file)
                print(len(all_file))

                print(all_file[23311])
                lag = [(read_file(file)) for file in all_file]
                # print(lag[1])
                print(len(lag))
                for  i in range(25000):
                   if(len(lag[i])!= 200):
                       print(i)
                lag = np.array(lag, dtype='float')
                print(lag.shape)
                print(all_file)
                print(len(all_file), lag.shape, frames)
                lag = lag.reshape(len(all_file), frames)  # (25000,100,2)->(25000,200)
                print(len(all_file), lag.shape, frames)
                # msdpd
                for i in range(25000):
                    MSDPD[i, 0] = label
                    msdpd(lag[i, :], MSDPD[i, 1:], frames)
                np.save(fea_file, MSDPD)
