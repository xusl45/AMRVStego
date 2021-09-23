# -*-coding:utf-8 -*-

'''
Implementation of our CNN-LSTM algorithm for VoIP
-------------
Based on paper:
    Steganalysis of VoIP Streams with CNN-LSTM Network
-------------
Author: Hao Yang
Email:  yanghao17@mails.tsinghua.edu.cn
'''

import numpy as np
import os, random, pickle, csv, sys
from keras.layers import Reshape, Dense, Dropout, LSTM, Bidirectional, AveragePooling1D, Flatten, Input, MaxPooling1D, \
    Convolution1D, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Activation
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate
from sklearn.metrics import recall_score

np.set_printoptions(threshold=np.inf)
model_path = "./model/weights_cnn_100_DN.h5"  # The path of model weights
SAMPLE_LENGTH = 1000  # The sample length (ms)
BATCH_SIZE = 256  # batch size
ITER = 200  # number of iteration
FOLD = 5  # = NUM_SAMPLE / number of testing samples
NUM_FILTERS = 64
FILTER_SIZES = [5, 4, 3]
num_class = 2
hidden_num = 64
k = 2


TRAIN_FOLDERS = [
    {"class": 1, "folder": "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\Chinese\\100"},
    {"class": 1, "folder": "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\English\\100"},
    {"class": 0, "folder": "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\Chinese\\cover"},
    {"class": 0, "folder": "D:\\DataSet\\Vstego_Fea\\Train\\DN-QIM\\English\\cover"}
]
TEST_FOLDERS = [
    {"class": 1, "folder": "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\Chinese\\100"},
    {"class": 1, "folder": "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\English\\100"},
    {"class": 0, "folder": "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\Chinese\\cover"},
    {"class": 0, "folder": "D:\\DataSet\\Vstego_Fea\\Test\\DN-QIM\\English\\cover"}
]

'''
Get the paths of all files in the folder
'''


def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


'''
Read codeword file
-------------
input
    file_path
        The path to an ASCII file.
        Each line contains three integers: x1 x2 x3, which are the three codewords of the frame.
        There are (number of frame) lines in total.
output
    the list of codewords
'''


def parse_sample(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    array = []
    for line in lines:
        row = line.strip("\r\n\t").strip().split(" ")
        row = list(map(float, row))
        array.append(row)
    file.close()
    return array


'''
Save variable in pickle
'''


def save_variable(file_name, variable):
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


if __name__ == '__main__':
    random.seed(777)

    train_files = [(item, folder["class"]) for folder in TRAIN_FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(train_files)
    train_samples_x = [(parse_sample(item[0])) for item in train_files]
    train_samples_y = [item[1] for item in train_files]
    np_train_samples_x = np.asarray(train_samples_x)
    np_train_samples_y = np.asarray(train_samples_y)
    file_num = len(train_files)
    sub_file_num = int(file_num / 5)
    x_val = np_train_samples_x[0: sub_file_num]
    y_val_ori = np_train_samples_y[0: sub_file_num]
    x_train = np_train_samples_x[sub_file_num: file_num]
    y_train_ori = np_train_samples_y[sub_file_num: file_num]

    test_files = [(item, folder["class"]) for folder in TEST_FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(test_files)
    test_samples_x = [(parse_sample(item[0])) for item in test_files]
    test_samples_y = [item[1] for item in test_files]
    x_test = np.asarray(test_samples_x)
    y_test_ori = np.asarray(test_samples_y)

    y_train = np_utils.to_categorical(y_train_ori, num_classes=2)
    y_val = np_utils.to_categorical(y_val_ori, num_classes=2)
    y_test = np_utils.to_categorical(y_test_ori, num_classes=2)

    print("Building model")
    in_shape = x_train.shape[1:]
    model_input = Input(shape=in_shape)
    rnn_feature = Bidirectional(LSTM(hidden_num, return_sequences=True))(model_input)
    pooled_outputs = []
    for filter_size in FILTER_SIZES:
        x = Conv1D(NUM_FILTERS * 2, filter_size, padding='same')(rnn_feature)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = AveragePooling1D(int(x.shape[1]))(x)
        x = GlobalAveragePooling1D()(x)
        x = Reshape((1, -1))(x)
        pooled_outputs.append(x)

    merged = concatenate(pooled_outputs)
    x = Flatten()(merged)
    x = Dropout(0.6)(x)
    x = Dense(NUM_FILTERS)(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    model_output = Dense(num_class, activation='softmax')(x)
    model = Model(model_input, model_output)

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    print("Training")
    for i in range(ITER):
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_val, y_val))
    model.save_weights(model_path)
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict, axis=1)
    print('* accuracy on test set: %0.2f%%' % (compute_accuracy(y_test_ori, y_predict) * 100))
    # y_predict = (y_predict > 0.5)
    tpr = recall_score(y_test_ori, y_predict)
    tnr = recall_score(y_test_ori, y_predict, pos_label=0)
    fpr = 1 - tnr
    fnr = 1 - tpr
    print('* FPR on test set: %0.2f' % (fpr * 100))
    print('* FNR on test set: %0.2f' % (fnr * 100))
    f = open("result_dn.txt", 'a')
    f.writelines(["\n" + model_path + " Accuracy %0.2f;" % (
            compute_accuracy(y_test_ori, y_predict) * 100) + " FPR %0.2f;" % (fpr * 100) + "FNR %0.2f" % (fnr * 100)])
    f.close()
