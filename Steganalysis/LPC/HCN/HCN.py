#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
Implementation of our CNN-LSTM algorithm for VoIP
-------------
Based on paper:
    Hierarchical Representation Network for Steganalysis of QIM Steganography in Low-Bit-Rate Speech Signals
-------------
"""

import time
import numpy as np
import os, random, pickle, csv, sys
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization, Activation, Concatenate, Conv1D, AdditiveAttention
from sklearn.metrics import recall_score
from keras_self_attention import SeqSelfAttention

model_path = "./1.0s/weights_net_100.h5"  # The path of model weights
BATCH_SIZE = 256  # batch size
ITER = 200  # number of iteration
FOLD = 5  # = NUM_SAMPLE / number of testing samples
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

FOLDERS = [
    {"class": 0, "folder": "D:\\AudioData\\AMR_NB_COVER_LSP\\Chinese\\1.0s"},
    {"class": 0, "folder": "D:\\AudioData\\AMR_NB_COVER_LSP\\English\\1.0s"},
    {"class": 1, "folder": "D:\\AudioData\\AMR_NB_STEGO_LSP\\Chinese\\1.0s\\100"},
    {"class": 1, "folder": "D:\\AudioData\\AMR_NB_STEGO_LSP\\English\\1.0s\\100"}
]

'''
Get the paths of all files in the folder
'''


def get_file_list(folder):
    file_list = []
    flag = 0
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
        flag += 1
        if flag == 20000:
            break
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


def HCN(input_shape):
    model_input = Input(shape=input_shape)

    x = Conv1D(256, 1, activation="relu", padding="same")(model_input)
    a1 = AdditiveAttention()([x, x])

    x = Conv1D(256, 3, activation="relu", padding="same")(x)
    a2 = AdditiveAttention()([x, x])

    x = Conv1D(256, 5, activation="relu", padding="same")(x)
    a3 = AdditiveAttention()([x, x])

    x = Concatenate()([a1, a2, a3])
    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(2, activation='softmax')(x)
    return Model(model_input, x)


if __name__ == '__main__':
    random.seed(777)

    all_files = [(file_path, folder["class"]) for folder in FOLDERS for file_path in get_file_list(folder["folder"])]
    random.shuffle(all_files)

    all_samples_x = [(parse_sample(item[0])) for item in all_files]
    all_samples_y = [item[1] for item in all_files]

    np_all_samples_x = np.asarray(all_samples_x)
    np_all_samples_y = np.asarray(all_samples_y)

    file_num = len(all_files) - 4000
    sub_file_num = int(file_num / 5)

    x_val = np_all_samples_x[0: sub_file_num]
    y_val_ori = np_all_samples_y[0: sub_file_num]

    x_train = np_all_samples_x[sub_file_num: file_num]
    y_train_ori = np_all_samples_y[sub_file_num: file_num]

    x_test = np_all_samples_x[file_num:]
    y_test_ori = np_all_samples_y[file_num:]

    y_train = np_utils.to_categorical(y_train_ori, num_classes=2)
    y_val = np_utils.to_categorical(y_val_ori, num_classes=2)
    y_test = np_utils.to_categorical(y_test_ori, num_classes=2)

    print("Building model")
    in_shape = x_train.shape[1:]
    model = HCN(in_shape)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True,
                                 mode='min', save_weights_only=True)
    callbacks_list = [checkpoint]
    print("Training")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=ITER, validation_data=(x_val, y_val), callbacks=callbacks_list)
    model.load_weights(model_path)
    start = time.time()
    y_predict = model.predict(x_test)
    end = time.time()
    y_predict = np.argmax(y_predict, axis=1)
    print('* accuracy on test set: %0.2f%%' % (compute_accuracy(y_test_ori, y_predict) * 100))
    # y_predict = (y_predict > 0.5)
    tpr = recall_score(y_test_ori, y_predict)
    tnr = recall_score(y_test_ori, y_predict, pos_label=0)
    fpr = 1 - tnr
    fnr = 1 - tpr
    print('* FPR on test set: %0.2f' % (fpr * 100))
    print('* FNR on test set: %0.2f' % (fnr * 100))
    f = open("result.txt", 'a')
    f.writelines(["\n" + model_path + " Accuracy %0.2f  " % (
            compute_accuracy(y_test_ori, y_predict) * 100) + "FPR %0.2f  " % (fpr * 100) + "FNR %0.2f  " % (fnr * 100)])
    f.close()
