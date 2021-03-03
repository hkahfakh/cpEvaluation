# coding:utf-8
import numpy as np
from math import pow
from sklearn import metrics
from multiprocessing import Pool as ProcessPool


def condition_ent_thread(arg):
    HZYX = 0.0
    number, intersection, condition, z = arg[0], arg[1], arg[2], arg[3]

    for i in intersection:
        c = condition[0] == i[0]
        for t in range(1, len(i)):  # 把所有条件与一下   就是同时满足一项条件的位置了
            c = np.bitwise_and(c, (condition[t] == i[t]))
        sub_z = z[c]
        if sub_z.size != 0:
            label_idx = np.unique(sub_z, return_inverse=True)[1]
            pi = np.bincount(label_idx).astype(np.float64)
            HZYX += -np.sum((pi / number) * (sub_z.shape[0] / number) * np.log(pi / number))
    return HZYX


def entropy(labels):
    """
    Calculates the entropy for a labeling.
    Parameters
    ----------
    labels : int array, shape = [n_samples]
        The labels
    """
    if len(labels) == 0:
        return 1.0

    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)  # 总数
    # log(a / b) should be calculated as log(a) - log(b) for possible loss of precision
    # 直接数组每一位都除pi_sum，变成了概率
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))


def condition_ent(x, y):
    """
    calculate ent H(y|x)
    计算条件信息熵  用概率直接算的
    或许用后面这个会好些    H(Y|X) = H(X,Y) - H(X)
    :param x:
    :param y:
    :return:
    """
    x_value_list, label_idx = np.unique(x, return_inverse=True)
    pi = np.bincount(label_idx).astype(np.float64)
    temp_ent = list()

    for x_value in x_value_list:
        sub_y = np.array(y)[x == x_value]
        temp_ent.append(entropy(sub_y))  # 计算当一个指定的x时，y的熵
    ent = (pi / np.sum(pi)) * temp_ent
    return ent.sum()


def two_condition_ent(x, y, z):
    HZYX = 0.0

    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    xylen = x.shape[0] * y.shape[0]
    for x_value in x_value_list:
        # print("1")  # 证明程序没有被卡死
        for y_value in y_value_list:
            sub_z = z[(x == x_value) & (y == y_value)]
            if sub_z.shape[0] != 0:
                temp_ent = entropy(sub_z)  # 计算当一个指定的x时，y的熵
                HZYX += (float(sub_z.shape[0]) / xylen) * temp_ent
    return HZYX


def condition_ent_plus(condition, z):
    """
    多个条件的条件熵
    :param condition:里面都是一个个特征
    :param z:
    :return:
    """
    # if condition == z:
    #     return entropy(z)
    HZYX = 0.0

    condition_value = [np.unique(condition[i]) for i in range(len(condition))]  # 求出了每个子数组有哪些元素
    intersection = np.array(np.meshgrid(*condition_value)).T.reshape(-1, len(condition_value))
    number = pow(len(condition[0]), len(condition))  # 概率的分母

    for i in intersection:
        c = condition[0] == i[0]
        for t in range(1, len(i)):
            c = c & (condition[t] == i[t])
        sub_z = z[c]
        if sub_z.shape[0] != 0:
            temp_ent = entropy(sub_z)  # 计算当一个指定的x时，y的熵
            HZYX += (float(sub_z.shape[0]) / number) * temp_ent

    return HZYX


def joint_entropy(x, y):
    """
    联合熵等于熵加条件熵
    :param x:
    :param y:
    :return:
    """
    HX = entropy(x)
    HYX = condition_ent(y, x)
    return HX + HYX


def joint_entropy_three(x, y, z):
    """
    熵的链式法则计算
    H(x1,x2,x3)=H(x1)+H(x2|x1)+H(x3|x2x1)
    :param,x x:
    :param y:
    :param z:
    :return:
    """
    HX = entropy(x)
    HYX = condition_ent(y, x)
    HZYX = 0.0

    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    xylen = x.shape[0] * y.shape[0]
    for x_value in x_value_list:
        # print("1")  # 证明程序没有被卡死
        for y_value in y_value_list:
            sub_z = z[(x == x_value) & (y == y_value)]
            if sub_z.shape[0] != 0:
                temp_ent = entropy(sub_z)  # 计算当一个指定的x时，y的熵
                HZYX += (float(sub_z.shape[0]) / xylen) * temp_ent

    return HX + HYX + HZYX


def condition_MI(x, y, z):
    """
    条件互信息
    MI(X1;X2|Y)=H(X1，Y)+H(X2，Y)-H(Y)-H(X1，X2，Y)
    :param x:
    :param y:
    :param z:
    :return:
    """
    HXZ = joint_entropy(x, z)
    HYZ = joint_entropy(x, z)
    HZ = entropy(z)
    HXYZ = joint_entropy_three(x, y, z)
    return HXZ + HYZ - HZ - HXYZ


def MI_chain(fi, F, S):
    """
    互信息的链式法则
    :param fi: 待选特征
    :param F: 特征集合 和后面的下标集合配合使用
    :param S: 已选特征 存了一条条特征的数据  这是个存特征原始数组中下标的列表  是要靠self.S传进来的
    :return:
    """
    H1 = entropy(F[:, S[0]])
    H2 = condition_ent(fi, F[:, S[0]])

    for i in range(1, len(S)):  # 就一个已选特征的话没必要进循环
        a = F[:, S[:i]]
        a = a.reshape(a.shape[1], -1)
        H1 = H1 + condition_ent_plus(a, F[:, S[i]])

    for i in range(1, len(S)):
        a = F[:, S[:i]]
        a = a.reshape(a.shape[1], -1)
        c = np.row_stack((a, fi))  # 把待选特征加到条件里面
        H2 = H2 + condition_ent_plus(c, F[:, S[i]])
    return H1 - H2


if __name__ == '__main__':
    x = np.ones(5, dtype=int)
    y = np.ones(5, dtype=int)
    z = np.ones(5, dtype=int)
