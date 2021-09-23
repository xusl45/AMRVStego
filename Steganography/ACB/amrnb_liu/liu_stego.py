#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: wu-junyan
@file: liu_stego.py
@time: 2021/2/28 0:18
@desc: n=4
"""
import os, math
from tqdm import tqdm
import numpy as np


def createDir(path):
    path = path.strip()
    path = path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def m4(period, m):
    a = np.random.randint(2, size=period)
    for i in range(period * period - 1):
        m[i] = a[0]
        # temp = (a[0]+a[3])%2
        temp = (a[0] + a[1]) % 2
        a[0] = a[1]
        a[1] = a[2]
        a[2] = a[3]
        a[3] = temp


# 还没写成功这个函数，先用m3吧
# 生成m序列https://blog.csdn.net/weixin_39833030/article/details/99718054
def mseq(period, mseq):
    # 多项式指数https://wenku.baidu.com/view/b11ca06b844769eae109ed32.html
    global fbconnection
    if period == 2:
        fbconnection = ["1", "1"]  # 多项式指数
    elif period == 3:
        fbconnection = ["1", "0", "1"]  # 多项式指数
    elif period == 4:
        fbconnection = ["1", "0", "0", "1"]
    register = np.zeros(period)
    newregister = np.zeros(period)
    newregister[period - 1] = 1
    mseq[0] = register[0]
    for i in range(1, int(math.pow(2, period) - 2)):
        tmp = 0
        for j in range(period):
            tmp = tmp + int(fbconnection[j]) * int(register[j])
        newregister[0] = tmp % 2
        for j in range(1, period - 1):
            newregister[j] = register[j - 1]
        register = newregister
        mseq[i - 1] = register[period - 1]


def Binary_idx(num, j):
    if j % 2 == 0:
        return int(abs(num / 2))
    elif j % 2 == 1:
        return int(abs(num % 2))


def PSV(a, b, period):  # 求两个序列相同元素的个数
    cc = 0
    for i in range(period):
        if a[i] == b[i]:
            cc = cc + 1
    return cc


def Re_Binary_idx(a, sign):
    num = 2 * a[0] + 1 * a[1]
    if sign == 1:
        num = -num
    return num


period = 4
times = ["0.1"]
emrs = ["10","20","40","50","100"]
n1, n2 = 0, 0

# https://www.cnblogs.com/ruanjianxian/p/6126839.html
if __name__ == '__main__':
    for time in times:
        for emr in emrs:
            time_f = float(time)
            emr_i = int(emr)
            # S1_path = "Z:/AMR_NB_STEGO/liu/%ss/%s/s1" % (time, emr)
            # createDir(S1_path)
            # print(S1_path)
            for filenu in range(8000):
                # 导入小数参数
                total_subframes = int(50 * time_f * 4)  # 总子帧数
                lagf = np.loadtxt("Z:/AMR_NB_FEA/liu/%ss/0/AMRNB_sample%04d.txt" % (time, filenu), dtype=int)
                lagf = lagf[:, -4:].reshape(total_subframes)
                lenB = total_subframes * 3  # 一帧用3bit表示
                print("小数:", lagf)
                # 生成sign flag
                sign = np.zeros(total_subframes, dtype=int)
                for i in range(total_subframes):
                    sign[i] = 0 if lagf[i] >= 0 else 1
                # 生成M序列
                # S1_path_file = S1_path + "/s1_%04d.txt" % filenu
                S1 = np.zeros((period * period - 1), dtype=int)  # 长度为2^(M周期)-1
                m4(period, S1)
                print("M序列S1长度", len(S1), ",周期", period, ",S1:", S1)
                print("嵌入率为", emr_i, "需要改变", int(emr_i * total_subframes / 100), "/", total_subframes, "个子帧")
                # # 生成标识用于指定某子帧进行嵌入
                insert = np.zeros(int(total_subframes / 2), dtype=int)
                for i in range(int(emr_i * total_subframes / 100 / 2)):  # 1秒生成100个奇数子帧嵌入标识,代表这个奇数子帧和下个子帧是否用于嵌入
                    insert[i] = 1
                np.random.shuffle(insert)  # 随机打乱
                print("嵌入标识", insert, "嵌入标识维度", insert.shape, "嵌入标识1的个数", np.sum(insert == 1))
                for i, flag in enumerate(insert):
                    if flag == 1:  # 执行嵌入
                        # 将该小数转为2进制（3bit,第一bit为符号位）用于嵌入秘密信息
                        # 生成秘密信息
                        print("--------------修改第", i * 2 + 1, "子帧:", lagf[i * 2], ",和第", i * 2 + 2, "子帧:", lagf[i * 2 + 1], "--------------")
                        secretM = np.random.randint(2, size=period)
                        s1 = np.zeros(period, dtype=int)
                        encryptedM = np.zeros(period, dtype=int)
                        Carryb = np.zeros(period, dtype=int)
                        for j in range(period):
                            Carryb[j] = Binary_idx(lagf[i * 2], j) if j < 2 else Binary_idx(lagf[i * 2 + 1], j)
                        for j in range(period):
                            s1[j] = S1[(j + i * period) % 15]
                            # s2[j] = S2[(j + i * period) % 15]
                            encryptedM[j] = secretM[j] ^ s1[j]  # 加密
                        print("小数二进制载体：", Carryb, "秘密信息", secretM, "加密后:", encryptedM, "符号位:", sign[i * 2], sign[i * 2 + 1])
                        psv = PSV(Carryb, encryptedM, period)
                        # 0 ≤ n1 ≤ n2 ≤ 8(len) , psv < n1 不换即原始基音延迟B , psv >= n2 替换即秘密信息M , psv < n2 则（1 - s2）*B + s2 * M
                        if psv >= n2:
                            for j in range(period):
                                Carryb = encryptedM
                        elif psv >= n1 and psv < n2:
                            if S2[i / period] == 1:
                                for j in range(period):
                                    Carryb = encryptedM
                        lagf[i * 2] = Re_Binary_idx(Carryb[0:2], sign[i * 2])
                        lagf[i * 2 + 1] = Re_Binary_idx(Carryb[2:4], sign[i * 2 + 1])
                        print("相似度:", psv, "修改后:", Carryb, lagf[i * 2], lagf[i * 2 + 1])
                print("修改后的小数:", lagf)
                createDir("Z:/AMR_NB_FEA/liu/%ss/%s/" % (time, emr))
                lagf = np.savetxt("Z:/AMR_NB_FEA/liu/%ss/%s/py_lagf%04d.txt" % (time, emr, filenu), lagf, fmt="%d", delimiter=" ")
