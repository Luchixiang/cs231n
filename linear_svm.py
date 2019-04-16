import numpy as np
from random import shuffle
"""
Wmis weight X is small 
X is minibatch
y is labels
reg is regularization strength

"""


def svm_loss_navie(W, X, y, reg):
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                #  梯度j==yi 为 xit  y!=yi 为-xit
                dW[:, j] += X[i].T
                dW[:, y[i]] += -X[i].T
    loss = loss/num_train
    dW /= num_train
    loss = loss + 0.5*reg*np.sum(W*W)
    dW = dW + reg * W
    return loss, dW