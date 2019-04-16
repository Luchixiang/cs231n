import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]
    diff = np.tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff**2
    squarDistance = np.sum(sqdiff, axis=1)
    realDistance = squarDistance**0.5
    sortedDisIndex = np.argsort(realDistance)  # 返回下标
    classCount = {}
    for i in range(k):
        voteLable = label[sortedDisIndex[i]]
        classCount[voteLable] = classCount.get(voteLable, 0) + 1
    maxcount = 0
    for key, value in classCount.items():
        if value > maxcount:
            maxcount = value
            classes = key
    return classes


dataSet, label = createDataSet()
input = np.array([1.1, 0.3])
output = classify(input, dataSet, label, 3)
print(output)
