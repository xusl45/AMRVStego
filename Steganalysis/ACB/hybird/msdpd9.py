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


def get_file_list_clag(folder):
    file_list = []
    for file in os.listdir(folder):
        if file[:5] == "clagf":
            #if file[:5] != "clagf":
                file_list.append(os.path.join(folder, file))
    return file_list


def savepkl(path, data):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


def msdpd9(lag, msdpd, frames):
    so = np.zeros(frames)
    ms = np.zeros(3)
    zy = np.zeros((3, 3))
    for i in range(frames - 2):
        so[i] = lag[i + 2] - 2 * lag[i + 1] + lag[i]
    for i in range(frames - 2):
        for j in range(-1, 2, 1):
            if so[i] == j:
                ms[j + 1] = ms[j + 1] + 1
    for i in range(frames - 2):
        for l in range(-1, 2, 1):
            for k in range(-1, 2, 1):
                if so[i] == l:
                    if so[i + 1] == k:
                        zy[l + 1][k + 1] = zy[l + 1][k + 1] + 1
    k = 0
    for i in range(3):
        for j in range(3):
            msdpd[k] = float(zy[i][j]) / ms[i]
            if ms[i] == 0:
                msdpd[k] = 0
            k = k + 1


if __name__ == '__main__':
    times = ["1"]  # "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
    emrs = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]  #"0", "10", "20", "30", "40", "50",
    stegos = ["yan2",'huang',"liu"]
    for time in times:
        for emr in emrs:
            for stego in stegos:
                time_f = float(time)
                frames = int(time_f * 200)
                MSDPD9 = np.zeros((25000, 10))  # label(1) + msdpd9(9)
                lag_path_chinese = "D:\\Vstego\\ACB\\int\\%s\\Chinese\\%s" % (stego, emr)  # 存放基因延迟参数的路径
                lag_path_english = "D:\\Vstego\\ACB\\int\\%s\\English\\%s" % (stego, emr)  # 存放基因延迟参数的路径
                lag_path_chinese_test = "D:\\Vstego\\testACB\\int\\%s\\Chinese\\%s" % (stego, emr)  # 存放基因延迟参数的路径
                lag_path_english_test = "D:\\Vstego\\testACB\\int\\%s\\English\\%s" % (stego, emr)  # 存放基因延迟参数的路径
                fea_file = "D:\\Vstego\\ACB\\npy\\msdpd9_%ss_%s_%s.npy" % (time, emr, stego)
                print("load data from ", lag_path_chinese, lag_path_chinese_test, lag_path_english,
                      lag_path_english_test, "\tsave at ", fea_file)
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
                all_file = lag_file_English + lag_file_chinese + lag_file_English_test + lag_file_chinese_test
                print(all_file)
                print(len(all_file))
                lag = [(read_file(file)) for file in all_file]
                # print(lag[1])
                print(len(lag))

                lag = np.array(lag, dtype='float').reshape(len(all_file), frames)  # (8000,100,2)->(8000,200)
                # msdpd9
                for i in range(25000):
                    MSDPD9[i, 0] = label
                    msdpd9(lag[i, :], MSDPD9[i, 1:], frames)
                np.save(fea_file, MSDPD9)
