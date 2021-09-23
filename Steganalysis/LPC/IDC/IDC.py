#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Implementation of IDC algorithm
-------------
Based on paper:
    Detection of quantization index modulation steganography in G. 723.1 bit stream based on quantization index sequence analysis
-------------
Author: Zinan Lin
Email:  zinanl@andrew.cmu.edu
'''

import os, random, pickle, csv, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from tqdm import tqdm

FOLD = 3  # = NUM_SAMPLE / number of testing samples
NUM_SAMPLE = 20000  # total number of samples used for training
TEST_NUM_SAMPLE = 5000  # number of samples used for test
'''
IDC feature extraction
-------------
input
    file
        The path to an ASCII file.
        Each line contains three integers: x1 x2 x3, which are the three codewords of the frame.
        There are (number of frame) lines in total. 
output
    A numpy vector, which contains the features determined by IDC algorithm.
'''


def IDC(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            line = [int(i) for i in line.split()]
            data.append(line)

    a = np.zeros(shape=(128, 128))
    e = np.zeros(shape=(256, 256))
    l = np.zeros(shape=(512, 512))
    m = np.zeros(shape=(256, 256))
    n = np.zeros(shape=(64, 64))
    c1 = np.zeros(shape=128)
    c2 = np.zeros(shape=256)
    c3 = np.zeros(shape=512)
    c4 = np.zeros(shape=256)
    c5 = np.zeros(shape=64)

    for i in range(len(data) - 1):
        data1 = data[i]
        data2 = data[i + 1]
        c1[data1[0]] += 1
        c2[data1[1]] += 1
        c3[data1[2]] += 1
        c4[data1[3]] += 1
        c5[data1[4]] += 1
        a[data1[0], data2[0]] += 1
        e[data1[1], data2[1]] += 1
        l[data1[2], data2[2]] += 1
        m[data1[3], data2[3]] += 1
        n[data1[4], data2[4]] += 1

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if c1[i] != 0:
                a[i, j] /= c1[i]
    for i in range(e.shape[0]):
        for j in range(e.shape[1]):
            if c2[i] != 0:
                e[i, j] /= c2[i]
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            if c3[i] != 0:
                l[i, j] /= c3[i]
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if c4[i] != 0:
                m[i, j] /= c4[i]
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            if c5[i] != 0:
                n[i, j] /= c5[i]

    c1 = c1 / (len(data) - 1)
    c2 = c2 / (len(data) - 1)
    c3 = c3 / (len(data) - 1)
    c4 = c4 / (len(data) - 1)
    c5 = c5 / (len(data) - 1)
    a = a.max(axis=1)
    e = e.max(axis=1)
    l = l.max(axis=1)
    m = m.max(axis=1)
    n = n.max(axis=1)

    return np.concatenate([c1, c2, c3, c4, c5, a, e, l, m, n])


'''
IDC training and testing
-------------
input
    positive_data_folder
        The folder that contains positive data files.
    negative_data_folder
        The folder that contains negative data files.
    result_folder
        The folder that stores the results.
'''


def main(positive_data_folder, negative_data_folder, t_positive_data_folder, t_negative_data_folder, result_folder):
    build_model = IDC

    positive_data_files = []
    for path in os.listdir(positive_data_folder[0]):
        positive_data_files.append(os.path.join(positive_data_folder[0], path))
    for path in os.listdir(positive_data_folder[1]):
        positive_data_files.append(os.path.join(positive_data_folder[1], path))

    negative_data_files = []
    for path in os.listdir(negative_data_folder[0]):
        negative_data_files.append(os.path.join(negative_data_folder[0], path))
    for path in os.listdir(negative_data_folder[1]):
        negative_data_files.append(os.path.join(negative_data_folder[1], path))

    t_positive_data_files = []
    for path in os.listdir(t_positive_data_folder[0]):
        t_positive_data_files.append(os.path.join(t_positive_data_folder[0], path))
    for path in os.listdir(t_positive_data_folder[1]):
        t_positive_data_files.append(os.path.join(t_positive_data_folder[1], path))

    t_negative_data_files = []
    for path in os.listdir(t_negative_data_folder[0]):
        t_negative_data_files.append(os.path.join(t_negative_data_folder[0], path))
    for path in os.listdir(t_negative_data_folder[1]):
        t_negative_data_files.append(os.path.join(t_negative_data_folder[1], path))

    train_positive_data_files = positive_data_files[0:NUM_SAMPLE]  # The positive samples for training
    train_negative_data_files = negative_data_files[0:NUM_SAMPLE]  # The negative samples for training

    test_positive_data_files = t_positive_data_files[0:TEST_NUM_SAMPLE]  # The positive samples for testing
    test_negative_data_files = t_negative_data_files[0:TEST_NUM_SAMPLE]  # The negative samples for testing

    num_train_files = len(train_negative_data_files)
    num_test_files = len(test_negative_data_files)

    # load train data
    print("Loading train data")
    X = []
    Y = []
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_negative_data_files[i])
        X.append(new_feature.reshape(1, -1))
        Y.append(0)
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_positive_data_files[i])
        X.append(new_feature.reshape(1, -1))
        Y.append(1)
    X = np.row_stack(X)

    # train SVM
    print("Training SVM")
    clf = svm.SVC()
    clf.fit(X, Y)
    with open(os.path.join(result_folder, "svm.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # test
    print("Testing")
    X = []
    Y = []
    for i in tqdm(range(num_test_files)):
        new_feature = build_model(test_negative_data_files[i])
        X.append(new_feature.reshape(1, -1))
        Y.append(0)
    for i in tqdm(range(num_test_files)):
        new_feature = build_model(test_positive_data_files[i])
        X.append(new_feature.reshape(1, -1))
        Y.append(1)
    X = np.row_stack(X)
    Y_predict = clf.predict(X)
    with open(os.path.join(result_folder, "Y_predict.pkl"), "wb") as f:
        pickle.dump(Y_predict, f)

    # output result
    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    print("Outputing result")
    with open(os.path.join(result_folder, "result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "real class", "predict class"])
        for i in range(num_test_files):
            writer.writerow([test_negative_data_files[i], 0, Y_predict[i]])
            if Y_predict[i] == 0:
                true_negative += 1
            else:
                false_positive += 1

        for i in range(num_test_files):
            writer.writerow([test_positive_data_files[i], 1, Y_predict[i + num_test_files]])
            if Y_predict[i + num_test_files] == 1:
                true_positive += 1
            else:
                false_negative += 1

        writer.writerow(["num of test files", 2 * num_test_files])
        writer.writerow(["True Positive", true_positive])
        writer.writerow(["True Negative", true_negative])
        writer.writerow(["False Positive", false_positive])
        writer.writerow(["False Negative", false_negative])
        writer.writerow(["Accuracy", float(true_negative + true_positive) / (num_test_files * 2)])
        writer.writerow(["Precision", float(true_positive) / (true_positive + false_positive)])
        writer.writerow(["Recall", float(true_positive) / (true_positive + false_negative)])
        writer.writerow(["FPR", float(false_positive) / (false_positive + true_negative)])
        writer.writerow(["FNR", float(false_negative) / (true_positive + false_negative)])


if __name__ == "__main__":
    random.seed(777)

    positive_data_folder = ["D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\Chinese\\100",
                            "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\English\\100"]
    negative_data_folder = ["D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\Chinese\\cover",
                            "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\English\\cover"]
    t_positive_data_folder = ["D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\Chinese\\100",
                              "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\English\\100"]
    t_negative_data_folder = ["D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\Chinese\\cover",
                              "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\English\\cover"]

    main(positive_data_folder, negative_data_folder, t_positive_data_folder,
         t_negative_data_folder, "D:\\PythonProject\\Vstego\\IDC\\result_dn\\100")
