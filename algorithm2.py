# coding:utf-8
import numpy as np
from featuresAlgorithm.mim import *
from algorithm1 import Banzhaf


def getFi(victory):
    """
    返回胜利准则最大的  那个特征的下标
    :param victory: numpy数组
    :return:
    """
    return np.argmax(victory)


def calcValue(F, C):
    """
    准则函数
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return: 返回普通特征选择方法每个特征的价值 numpy数组
    """
    m = mim(F, C)
    return m.calcMI()


def featureSelection(F, C, delta):
    """

    :param F:
    :param C:
    :param Pv: 算法1计算出的Banzhaf权利指数
    :param delta: 选择特征的个数
    :return: 选出来的特征下标
    """
    S = []
    k = 0
    while k < delta:
        J = calcValue(F, C)
        b = Banzhaf(F, C)
        Pv = b.banzhafPowerIndex()

        victory = J * Pv
        i = getFi(victory)
        S.append(F[:, i])
        F = np.delete(F, -1, axis=1)  # 删除已经选择的特征
        k = k + 1
    return S
