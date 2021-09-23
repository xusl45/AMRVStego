# coding=utf-8
import pickle

import numpy as np
import os


def eo(sz, eoff, frames):
    i, j, k, o, e = (0, 0, 0, 0, 0)
    eo0 = np.zeros(16)
    eo1 = np.zeros(2)
    eo2 = np.zeros(4)
    eo3 = np.zeros(8)
    eo4 = np.zeros(16)
    for i in range(frames - 2):
        if k == 4:
            k = 0
        if k % 2 == 0:
            if sz[i] % 2 == 0:
                eo0[0] = eo0[0] + 1
                if sz[i + 1] % 2 == 0:
                    eo0[1] = eo0[1] + 1
        elif k % 2 != 0:
            if sz[i] % 2 != 0:
                eo0[2] = eo0[2] + 1
                if sz[i + 1] % 2 != 0:
                    eo0[3] = eo0[3] + 1
        if k == 0:
            if sz[i] % 2 == 0:
                eo0[4] = eo0[4] + 1
                if sz[i + 2] % 2 == 0:
                    eo0[5] = eo0[5] + 1
            elif sz[i] % 2 != 0:
                eo0[6] = eo0[6] + 1
                if sz[i + 2] % 2 != 0:
                    eo0[7] = eo0[7] + 1
        if k == 1:
            if sz[i] % 2 == 0:
                eo0[8] = eo0[8] + 1
                if sz[i + 2] % 2 == 0:
                    eo0[9] = eo0[9] + 1
            elif sz[i] % 2 != 0:
                eo0[10] = eo0[10] + 1
                if sz[i + 2] % 2 != 0:
                    eo0[11] = eo0[11] + 1
        if i % 4 == 0:
            if sz[i] % 2 == 0:
                eo0[12] = eo0[12] + 1
                if i + 4 != frames:
                    if sz[i + 4] % 2 == 0:
                        eo0[13] = eo0[13] + 1
            elif sz[i] % 2 != 0:
                eo0[14] = eo0[14] + 1
                if i + 4 != frames:
                    if sz[i + 4] % 2 != 0:
                        eo0[15] = eo0[15] + 1
        k = k + 1
    for i in range(frames - 5):
        if sz[i] % 2 == 0:
            eo1[0] = eo1[0] + 1
            if sz[i + 1] % 2 == 0:
                eo2[0] = eo2[0] + 1
                if sz[i + 2] % 2 == 0:
                    eo3[0] = eo3[0] + 1
                    if sz[i + 3] % 2 == 0:
                        eo4[0] = eo4[0] + 1
                    elif sz[i + 3] % 2 != 0:
                        eo4[1] = eo4[1] + 1
                elif sz[i + 2] % 2 != 0:
                    eo3[1] = eo3[1] + 1
                    if sz[i + 3] % 2 == 0:
                        eo4[2] = eo4[2] + 1
                    elif sz[i + 3] % 2 != 0:
                        eo4[3] = eo4[3] + 1
            elif sz[i + 1] % 2 != 0:
                eo2[1] = eo2[1] + 1
                if sz[i + 2] % 2 == 0:
                    eo3[2] = eo3[2] + 1
                    if sz[i + 3] % 2 == 0:
                        eo4[4] = eo4[4] + 1
                    elif sz[i + 3] % 2 != 0:
                        eo4[5] = eo4[5] + 1
                elif sz[i + 2] % 2 != 0:
                    eo3[3] = eo3[3] + 1
                    if sz[i + 3] % 2 == 0:
                        eo4[6] = eo4[6] + 1
                    elif sz[i + 3] % 2 != 0:
                        eo4[7] = eo4[7] + 1
        elif sz[i] % 2 != 0:
            eo1[1] = eo1[1] + 1
            if sz[i + 1] % 2 == 0:
                eo2[2] = eo2[2] + 1
                if sz[i + 2] % 2 == 0:
                    eo3[4] = eo3[4] + 1
                    if sz[i + 3] % 2 == 0:
                        eo4[8] = eo4[8] + 1
                    elif sz[i + 3] % 2 != 0:
                        eo4[9] = eo4[9] + 1
                elif sz[i + 2] % 2 != 0:
                    eo3[5] = eo3[5] + 1
                    if sz[i + 3] % 2 == 0:
                        eo4[10] = eo4[10] + 1
                    elif sz[i + 3] % 2 != 0:
                        eo4[11] = eo4[11] + 1
            elif sz[i + 1] % 2 != 0:
                eo2[3] = eo2[3] + 1
                if sz[i + 2] % 2 == 0:
                    eo3[6] = eo3[6] + 1
                    if sz[i + 3] % 2 == 0:
                        eo4[12] = eo4[12] + 1
                    elif sz[i + 3] % 2 != 0:
                        eo4[13] = eo4[13] + 1
                elif sz[i + 2] % 2 != 0:
                    eo3[7] = eo3[7] + 1
                    if sz[i + 3] % 2 == 0:
                        eo4[14] = eo4[14] + 1
                    elif sz[i + 3] % 2 != 0:
                        eo4[15] = eo4[15] + 1
    for i in range(2):
        if eo1[i] == 0:
            eoff[i] = 0
        else:
            eoff[i] = eo2[i * 2] / eo1[i]
    j = 0
    for i in range(2, 6):
        if eo2[int(j / 2)] == 0:
            eoff[i] = 0
        else:
            eoff[i] = eo3[j] / eo2[int(j / 2)]
        j = j + 2
    j = 0
    for i in range(6, 14):
        if eo3[int(j / 2)] == 0:
            eoff[i] = 0
        else:
            eoff[i] = eo4[j] / eo3[int(j / 2)]
        j = j + 2


def read_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    content = []
    for line in lines:
        line = line.replace("\t", " ")
        line_split = line.strip("\r\n\t").strip().split(" ")
        content.append(line_split)
    return content


def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


def savepkl(path, data):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


if __name__ == '__main__':
    stegos = ["liu","huang","yan2"]
    # times = [ "0.1", "0.2","0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    times = ["1"]
    for stego in stegos:
        for time in times:
            for emr in range(0, 101, 10):
                time_f=float(time)
                frames = int(time_f * 200)
                PBP = np.zeros((25000, 15))  # label(1) + PBP(14)
                lag_path_chinese = "D:\\Vstego\\ACB\\int\\%s\\Chinese\\%d" % (stego, emr)  # 存放基因延迟参数的路径
                lag_path_english = "D:\\Vstego\\ACB\\int\\%s\\English\\%d" % (stego, emr)  # 存放基因延迟参数的路径
                lag_path_chinese_test = "D:\\Vstego\\testACB\\int\\%s\\Chinese\\%d" % (stego, emr)  # 存放基因延迟参数的路径
                lag_path_english_test = "D:\\Vstego\\testACB\\int\\%s\\English\\%d" % (stego, emr)  # 存放基因延迟参数的路径
                # fea_file = "D:\\paper1Data\\NB\\%s\\npy\\PBP_%ss_%d_%s.npy" % (stego,time, emr, stego)
                fea_file = "D:\\Vstego\\ACB\\npy\\PBP_%ss_%d_%s.npy" % (time, emr, stego)
                print("load data from ", lag_path_chinese,lag_path_chinese_test,lag_path_english,lag_path_english_test, "\tsave at ", fea_file)
                if emr == 0:
                    label = 1
                else:
                    label = -1
                lag_file_chinese = get_file_list(lag_path_chinese)  # train
                lag_file_English = get_file_list(lag_path_english)  # train
                lag_file_chinese_test = get_file_list(lag_path_chinese_test)  # test
                lag_file_English_test = get_file_list(lag_path_english_test)  # test
                print(len(lag_file_English))
                print(len(lag_file_English))
                print(len(lag_file_chinese_test))
                print(len(lag_file_English_test))
                all_file = lag_file_English + lag_file_chinese + lag_file_English_test + lag_file_chinese_test
                print(len(all_file), all_file)
                lag = [(read_file(file)) for file in all_file]
                lag = np.array(lag, dtype='float').reshape(len(all_file), frames)  # (8000,100,2)->(8000,200)
                # PBP
                for i in range(25000):
                    PBP[i, 0] = label
                    eo(lag[i, :], PBP[i, 1:], frames)
                np.save(fea_file, PBP)
