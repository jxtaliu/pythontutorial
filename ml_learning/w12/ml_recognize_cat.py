# -*- coding: utf-8 -*-
"""
Created by liuxiang(jxta.liu@gmail.com)

@author:liuxiang

"""

import numpy as np
import matplotlib.pyplot as plt
from ml_learning.w12.lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 将训练集的维度降低并转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    """
    sigmoid function
    :param z:
    :return:
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    init parameters
    :param dim:
    :return:
    """
    w = np.zeros(shape=(dim, 1))
    b = 0
    return (w, b)


def propagate(w, b, X, Y):
    """
    实现前向和后向传播的成本函数及其梯度
    :param w:
    :param b:
    :param X:
    :param Y:
    :return:
    """

    m = X.shape[1]
    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))

    # 反向传播
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grade = {
        "dw": dw,
        "db": db
    }
    return (grade, cost)


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    利用梯度下降法实现优化w,b
    :param w:
    :param b:
    :param X:
    :param Y:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    costs = []
    for i in range(num_iterations):
        grade, cost = propagate(w, b, X, Y)
        dw = grade["dw"]
        db = grade["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            # 打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i, cost))
    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }
    return (params, grads, costs)


def predict(w, b, X):
    """
    使用学习逻辑回归参数logistic(w, b)预测标签是0还是1
    :param w:
    :param b:
    :param X:
    :return:
    """
    m = X.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计算猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X)+b)
    for i in range(A.shape[1]):
        y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    assert (y_prediction.shape == (1, m))

    return y_prediction


def model(x_train, y_train, x_test, y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    实现逻辑回归模型
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    w, b = initialize_with_zeros(x_train.shape[0])
    params, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, x_train)

    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": y_prediction_test,
        "Y_prediciton_train": y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# 绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()



