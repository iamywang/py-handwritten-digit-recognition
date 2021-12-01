import numpy
from os import listdir
from tensorflow.keras import layers, models


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
    labels = numpy.zeros([nums], int)
    for i in range(nums):
        fpath = flist[i]
        digit = int(fpath.split('_')[0])
        labels[i] = digit
        sets[i] = txtToMatrix(path + '/' + fpath)
    return sets, labels


cnn = models.Sequential()


def cnnInit():
    # 第1层卷积，卷积核大小为3*3，32个
    cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    cnn.add(layers.MaxPooling2D((2, 2)))

    # 第2层卷积，卷积核大小为3*3，64个
    cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
    cnn.add(layers.MaxPooling2D((2, 2)))

    # 第3层卷积，卷积核大小为3*3，64个
    cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # 展开
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(64, activation='relu'))
    cnn.add(layers.Dense(10, activation='softmax'))

    cnn.summary()


# 训练集
train_sets, train_labels = readData('digits/trainingDigits')
train_sets = train_sets.reshape((1934, 32, 32, 1))

# 测试集
sets, labels = readData('digits/testDigits')
sets = sets.reshape((946, 32, 32, 1))


def cnnTrain(times):
    # 训练
    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    cnn.fit(train_sets, train_labels, epochs=times)

    # cnn.evaluate(train_sets, train_labels)


def cnnTest(times):
    # 预测
    cnn.evaluate(sets, labels)
    res = cnn.predict(sets)

    error_num = 0
    for i in range(len(res)):
        if int(numpy.argmax(res[i])) != labels[i]:
            error_num += 1
        # print("index: %d, predict: %d, actual: %d" % (i, int(numpy.argmax(res[i])), labels[i]))

    # 输出结果
    output = ""
    output += "训练轮次：" + str(times) + ", "
    output += "测试总数：" + str(len(sets)) + ", "
    output += "错误数：" + str(error_num) + ", "
    output += "错误率：" + str(error_num / float(len(sets))) + ", "
    output += "正确率：" + str(1 - error_num / float(len(sets)))
    print(output)


# cnn test
cnnInit()
cnnTrain(1)
cnnTest(1)

cnnTrain(1)
cnnTest(2)

cnnTrain(1)
cnnTest(3)

cnnTrain(1)
cnnTest(4)

cnnTrain(1)
cnnTest(5)

cnnTrain(5)
cnnTest(10)

cnnTrain(10)
cnnTest(20)

# cnnTrain(30)
# cnnTest(50)

# cnnTrain(50)
# cnnTest(100)
