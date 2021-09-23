import util
import os, random
import numpy as np
import time



#files for train
FOLDERS_train = [
    {"class": -1, "folder": "./FCB/Geiser-C-2772/3Dtest/train/rate" },  # The folder that contains positive data files.   stego
    {"class": 1, "folder": "./FCB/Geiser-C-2772/3Dtest/train/Original"}  # The folder that contains negative data files.  cover
]

#files for test
FOLDERS_test = [
    {"class": -1, "folder": "./FCB/Geiser-C-2772/3Dtest/test/rate", },  # The folder that contains positive data files.   stego
    {"class": 1, "folder": "./FCB/Geiser-C-2772/3Dtest/test/Original"}  # The folder that contains negative data files.  cover
]

#PATH_train = "I:/Python/AutoEncoder/FCB/Miao-4-C-2772/10s/train"
#PATH_test = "I:/Python/AutoEncoder/FCB/Miao-4-C-2772/10s/test"

PATH_train = "I:/Python/AutoEncoder/FCB/Geiser-C-2772/3Dtest/train"
PATH_test = "I:/Python/AutoEncoder/FCB/Geiser-C-2772/3Dtest/test"

# Network Parameters
n_input = 2772
iterations = 20000
#n_hidden = [62,31,15,7,3,2,1]
n_hidden = [1]



#创建神经网络类
# nodes为1*n矩阵，表示每一层有多少个节点，例如[3,4,5]
# 表示三层，第一层有3个节点，第二层4个，第三层5个
class nn():
    def __init__(self,nodes):#输入为神经网络结构
        self.layers = len(nodes)
        self.nodes = nodes
        self.u = 1.0 # 学习率
        self.W = list() # 权值
        self.B = list() # 偏差值
        self.values = list() # 层值
        self.error = 0 # 误差
        self.loss = 0 # 损失

        for i in range(self.layers-1):
            self.W.append(np.random.random((self.nodes[i], self.nodes[i+1])) - 0.5)         # 权值初始化，权重范围-0.5~0.5
            self.B.append(0)               # 偏差B值初始化
        for j in range(self.layers):
            # 节点values值初始化
            self.values.append(0)

#创建autoencoder类，可以看成是多个神经网络简单的堆叠而来
class autoencoder():
    def __init__(self):
        self.encoders = list()
    def add_one(self,nn):
        self.encoders.append(nn)

# 激活函数
def sigmod(x):
    return 1.0 / (1.0 + np.exp(-x))

# 前馈函数
def nnff(nn, x, y):
    layers = nn.layers   #网络层数
    numbers = x.shape[0]  #样本数
    # 赋予初值
    nn.values[0] = x  #输入样本
    for i in range(1, layers):
        nn.values[i] = sigmod(np.dot(nn.values[i - 1], nn.W[i - 1]) + nn.B[i - 1])  #编码
    # 最后一层与实际的误差
    nn.error = y - nn.values[layers - 1]
    nn.loss = 1.0 / 2.0 * (nn.error ** 2).sum() / numbers
    train_loss.append(nn.loss)
    return nn

def cb(deltas,layers):
    data = deltas[layers][0]
    for i in range(1, deltas[layers].shape[0]):
        data += deltas[layers][i]
    for j in range(deltas[layers].shape[0]):
        deltas[layers][j] = data / (deltas[layers].shape[0])
    return deltas[layers]

# BP函数
def nnbp(nn):
    layers = nn.layers
    # 初始化delta

    deltas = list()
    for i in range(layers):
        deltas.append(0)

    # 最后一层的delta为
    deltas[layers - 1] = -nn.error * nn.values[layers - 1] * (1 - nn.values[layers - 1])
    ###计算
    #deltas[layers - 1] = cb(deltas,layers-1)

    # 其他层的delta为
    for j in range(1, layers - 1)[::-1]:  # 倒过来
        deltas[j] = np.dot(deltas[j + 1], nn.W[j].T) * nn.values[j] * (1 - nn.values[j])
        #deltas[j] = cb(deltas, j)
    # 更新W值
    for k in range(layers - 1):
        nn.W[k] -= nn.u * np.dot(nn.values[k].T, deltas[k + 1]) / (deltas[k + 1].shape[0])
        nn.B[k] -= nn.u * deltas[k + 1] / (deltas[k + 1].shape[0])
    return nn

# 对神经网络进行训练
def nntrain(nn, x, y, iterations):
    for i in range(iterations):
        if i%10 == 0:
            print("%d" %(i/10))
        nnff(nn, x, y)
        nnbp(nn)
        endtime = time.clock()
        train_time.append((endtime - starttime) / 2000)
    return nn

# 建立autoencoder框架
def aebuilder(nodes):
    layers = len(nodes)
    ae = autoencoder()
    for i in range(layers - 1):
        ae.add_one(nn([nodes[i], nodes[i + 1], nodes[i]]))
    return ae


# 训练autoencoder
def aetrain(ae, x, interations):
    elayers = len(ae.encoders)
    train_or_tiny = 1
    for i in range(elayers):
        # 单层训练
        ae.encoders[i] = nntrain(ae.encoders[i], x, x, interations)
        # 单层训练后，获取该层中间层的值，作为下一层的训练
        nntemp = nnff(ae.encoders[i], x, x)
        x = nntemp.values[1]
    return ae

# def predict(nn,x):

#输出降维后的数据（feature，time，loss）
def out(aecomplete,y,featurepath,trainlosspath,traintimepath):
    featurefile = open(featurepath, 'w+')
    trainloss = open(trainlosspath, 'w+')
    traintime = open(traintimepath, 'w+')

    feature = aecomplete.values[1]
    for i in range(feature.shape[0]):#样本数
        featurefile.write(str(y[i][0]) + ' ')
        for j in range(feature.shape[1]):#特征
            featurefile.write(str(j+1) + ':'+ str(feature[i][j]) + " ")
        featurefile.write('\n')
    featurefile.close()

    for j in range(iter):
        trainloss.write(str(j+1) + ' ' + str(train_loss[j]) + '\n')
        traintime.write(str(j+1) + ' ' + str(train_time[j]) + '\n')
    trainloss.close()
    traintime.close()

#获取数据
def get_file_list(folder,classs,emrate):
    file_list = []
    if classs == 1:
        rate = "/"
    else:
        rate = str(emrate) + "/"
    pathh = folder + rate
    for file in os.listdir(pathh):   #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        file_list.append(os.path.join(pathh, file))  #os.path.join()：  将多个路径组合后返回
    return file_list

def readFile(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    for line in lines:
        sample = line.strip("\r\n\t").strip().split(" ")  #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        #  split() 方法通过指定分隔符对字符串进行分割并返回一个列表，默认分隔符为所有空字符，包括空格、换行(\n)、制表符(\t)等
        sample = [float(item) for item in sample]
    return sample

def autoEncoder(x,y,nodes,iterations):
    # 建立auto框架
    ae = aebuilder(nodes)
    # 设置部分参数
    # 训练
    ae = aetrain(ae, x, iterations)

    nodescomplete = np.array([n_input, _n_hidden, n_input])
    aecomplete = nn(nodescomplete)
    aecomplete = ae.encoders[0]
    return aecomplete


###############################################################################################################################
#
###############################################################################################################################
train_loss = list()  # 训练误差
train_time = list()  # 训练时间
#emrate from 10% to 100%
for rate in range(10,11):
    emrate = rate*10

    # timefile = open(timepath, 'w+')
    #读取文件
    all_train_files = [(item, folder["class"]) for folder in FOLDERS_train for item in get_file_list(folder["folder"], folder["class"],emrate)]
    all_test_files = [(item, folder["class"]) for folder in FOLDERS_test for item in get_file_list(folder["folder"], folder["class"],emrate)]  # all_files是数组，保存所有文件，元素为(item, folder["class"])，item保存文件名列表,folder["class"]保存对应的类别。

    random.shuffle(all_train_files)  # 随机排序
    random.shuffle(all_test_files)

    all_train_x = [(readFile(item[0])) for item in all_train_files]  # 文件路径
    all_train_y = [[item[1]] for item in all_train_files]  # 类别

    all_test_x = [(readFile(item[0])) for item in all_test_files]  # 文件路径
    all_test_y = [[item[1]] for item in all_test_files]  # 类别

    np_all_train_x = np.asarray(all_train_x)  # asarray将输入数据（列表的列表，元组的元组，元组的列表等）转换为数组
    np_all_train_y = np.asarray(all_train_y)

    np_all_test_x = np.asarray(all_test_x)
    np_all_test_y = np.asarray(all_test_y)

    for dim in n_hidden:
        _n_hidden = dim
        print('start emrate = %d, ' %emrate, 'dim = %d' %dim)

        nodes = [n_input, _n_hidden]
        #for iter in range(500,iterations,500):#测试多个迭代次数的情况，每次迭代次数+500
        for iter in range(1500,12000,500):
            starttime = time.clock()
            # 输出路径
            print('train, iter = %d' % iter)
            featurePATH = '/feature-train-' + str(n_input) + '-' + str(_n_hidden) + '-' + str(emrate) + '-' + str(iter) + '.txt'  # 降维后的特征
            trainlossPATH = '/train-loss-' + str(n_input) + '-' + str(_n_hidden) + '-' + str(emrate) + '-' + str(iter) + '.txt'  # 训练误差
            traintimePATH = '/train-time-' + str(n_input) + '-' + str(_n_hidden) + '-' + str(emrate) + '-' + str(iter) + '.txt'  # 训练时间
            autotrain=autoEncoder(np_all_train_x,np_all_train_y,nodes,iter)
            featurepath = PATH_train + featurePATH
            trainlosspath = PATH_train + trainlossPATH
            traintimepath = PATH_train + traintimePATH
            out(autotrain,np_all_train_y,featurepath,trainlosspath,traintimepath)

            train_loss.clear()
            train_time.clear()

            # #testclassification = list()
            # testclassification = sigmod(np.dot(np_all_test_x, autotrain.W[0]) + autotrain.B[0])
            # testclassification = testclassification.tolist()
            # classificationpath = 'I:/Python/AutoEncoder/FCB/Miao-4-C-2772/test/classification/feature-' + str(emrate)+ '-'+ str(iter) + '.txt'
            # classification = open(classificationpath,'w+')
            # classification.write(str(testclassification[i][0]) + ' ')
            # for i in range(len(testclassification)):
            #     classification.write(str(i+1) + ':' + str(np_all_test_y[i][0]) + ' ')
            # classification.write('\n')
            # classification.close()

            print('test, iter = %d' % iter)
            starttime = time.clock()
            autotest=autoEncoder(np_all_test_x,np_all_test_y,nodes,iter)
            featurepath = PATH_test + featurePATH
            trainlosspath = PATH_test + trainlossPATH
            traintimepath = PATH_test + traintimePATH
            out(autotest,np_all_test_y,featurepath,trainlosspath,traintimepath)

            train_loss.clear()
            train_time.clear()

print('finished!')


