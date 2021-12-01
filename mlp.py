import numpy
from os import listdir
from sklearn import neural_network


# 处理手写数字
def txtToMatrix(f):
    res = numpy.zeros([1024], int)
    fr = open(f)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            res[i * 32 + j] = lines[i][j]
    return res


# 读数据集
def readData(path):
    flist = listdir(path)
    nums = len(flist)
    sets = numpy.zeros([nums, 1024], int)
    labels = numpy.zeros([nums, 10], int)
    for i in range(nums):
        fpath = flist[i]
        digit = int(fpath.split('_')[0])
        labels[i][digit] = 1.0
        sets[i] = txtToMatrix(path + '/' + fpath)
    return sets, labels


def mlpTest(hidden, rate, iter, solver):
    # 训练神经网络
    train_sets, train_labels = readData('digits/trainingDigits')
    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(hidden,), activation='logistic', solver=solver,
                                       learning_rate_init=rate, max_iter=iter)
    print(mlp)
    mlp.fit(train_sets, train_labels)

    # 测试集实验
    sets, labels = readData('digits/testDigits')
    res = mlp.predict(sets)
    error_num = 0
    for i in range(len(sets)):
        if numpy.sum(res[i] == labels[i]) < 10:
            error_num += 1

    # 输出结果
    output = ""
    output += "隐层节点数：" + str(hidden) + ", "
    output += "学习率：" + str(rate) + ", "
    output += "总数：" + str(len(sets)) + ", "
    output += "错误数：" + str(error_num) + ", "
    output += "错误率：" + str(error_num / float(len(sets))) + ", "
    output += "正确率：" + str(1 - error_num / float(len(sets)))
    print(output)


# hidden test
# mlpTest(500, 0.0001, 1000, 'adam')
# mlpTest(1000, 0.0001, 1000, 'adam')
# mlpTest(1500, 0.0001, 1000, 'adam')
# mlpTest(2000, 0.0001, 1000, 'adam')

# learning test
mlpTest(100, 0.0001, 100000, 'sgd')
mlpTest(100, 0.001, 100000, 'sgd')
mlpTest(100, 0.01, 100000, 'sgd')
mlpTest(100, 0.1, 100000, 'sgd')
