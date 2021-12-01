import numpy
from os import listdir
from sklearn import neighbors


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
    labels = numpy.zeros([nums])
    for i in range(nums):
        fpath = flist[i]
        digit = int(fpath.split('_')[0])
        labels[i] = digit
        sets[i] = txtToMatrix(path + '/' + fpath)
    return sets, labels


def knnTest(k):
    # 构建KNN分类器
    train_sets, train_labels = readData('digits/trainingDigits')
    knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)
    knn.fit(train_sets, train_labels)

    # 测试集实验
    sets, labels = readData('digits/testDigits')
    res = knn.predict(sets)
    error_num = numpy.sum(res != labels)

    # 输出结果
    output = ""
    output += "邻居节点K=" + str(k) + ", "
    output += "总数：" + str(len(sets)) + ", "
    output += "错误数：" + str(error_num) + ", "
    output += "错误率：" + str(error_num / float(len(sets))) + ", "
    output += "正确率：" + str(1 - error_num / float(len(sets)))
    print(output)


knnTest(1)
knnTest(3)
knnTest(5)
knnTest(7)
