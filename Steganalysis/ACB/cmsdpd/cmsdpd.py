# coding=utf-8
import numpy as np

# cmsdpd169的
# times = ["0.1","0.2", "0.3", "0.4", "0.5", "0.6","0.7", "0.8", "0.9"]
stegos = ["yan2", "huang", "liu"]
times = ["1"]
for time in times:
    for emr in range(0, 101, 10):
        for stego in stegos:
            # msdpd_file = "D:\\paper1Data\\NB\\%s\\npy\\msdpd-_%ss_%d_%s.npy" % (stego,time, emr,stego)
            # msdpdr_file = "D:\\paper1Data\\NB\\%s\\npy\\msdpd-r_%ss_%d_%s.npy" % (stego, time, emr,stego)
            # cmsdpd_file = "D:\\paper1Data\\NB\\%s\\npy\\cmsdpd_%ss_%d_%s.npy" % (stego,time, emr, stego)
            msdpd_file = "D:\\Vstego\\ACB\\npy\\msdpd_%ss_%d_%s.npy" % ( time, emr, stego)
            msdpdr_file = "D:\\Vstego\\ACB\\npy\\msdpd-r_%ss_%d_%s.npy" % ( time, emr, stego)
            cmsdpd_file = "D:\\Vstego\\ACB\\npy\\cmsdpd_%ss_%d_%s.npy" % ( time, emr, stego)
            msdpd = np.load(msdpd_file)
            msdpdr = np.load(msdpdr_file)
            time_f = float(time)
            frames = int(time_f * 200)
            cmsdpd = np.zeros((25000, 170))
            if emr == 0:
                label = 1
            else:
                label = -1
            for i in range(25000):
                cmsdpd[i, 0] = label
                cmsdpd[i, 1:] = msdpd[i, 1:] - msdpdr[i, 1:]
            print(msdpd_file, msdpdr_file, cmsdpd_file)
            np.save(cmsdpd_file, cmsdpd)

# 1.先对正常生成的隐写样本(pcm->dat)提取的整数特征(txt)运行msdpd.py生成msdpd特征
# 2.再对隐写样本(dat)运行recompression.c生成再编码(dat->pcm->dat)后的样本和对应整数特征(txt)
# 3.对在编码后的整数特征运行msdpd.py生成msdpd-r特征
# 4.用msdpd特征-msdpd-r特征即cmsdpd特征

# # cmsdpd9的
# times = ["1"]  # "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"
# emrs = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
# stegos = ["huang","yan2","liu"]
# for time in times:
#     for emr in emrs:
#         for stego in stegos:
#             msdpd9_file = "D:\\Vstego\\ACB\\npy\\msdpd9_%ss_%s_%s.npy" % (time, emr, stego)
#             msdpd9r_file = "D:\\Vstego\\ACB\\npy\\msdpd9-r_%ss_%s_%s.npy" % ( time, emr, stego)
#             cmsdpd9_file = "D:\\Vstego\\ACB\\npy\\cmsdpd9_%ss_%s_%s.npy" % (time, emr, stego)
#             msdpd = np.load(msdpd9_file)
#             msdpdr = np.load(msdpd9r_file)
#             time_f = float(time)
#             frames = int(time_f * 200)
#             cmsdpd = np.zeros((25000, 10))
#             if emr == 0:
#                 label = 1
#             else:
#                 label = -1
#             for i in range(25000):
#                 cmsdpd[i, 0] = label
#                 cmsdpd[i, 1:] = msdpd[i, 1:] - msdpdr[i, 1:]
#             print(msdpd9_file, msdpd9r_file, cmsdpd9_file)
#             np.save(cmsdpd9_file, cmsdpd)
