# coding:utf-8
from sklearn import metrics
import numpy as np
import math
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool


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


class Rmifs:
    def __init__(self, F, C, test_X, test_y, m=3, beta=0.5):
        self.F = F
        self.C = C  # 标签
        self.test_X = test_X
        self.test_y = test_y
        self.S = list()  # 已选特征的下标
        self.m = m  # 要选取m个特征  最小为1
        self.beta = beta  # 论文上说rmifs的β与mifs的是倍数关系
        self.num_cores = 4

        self.PoolType = 'Process'
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            # self.num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(self.num_cores)  # 设置池的大小  为计算机核心数

    def entropy(self, labels):
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

    def condition_ent(self, x, y):
        """
            calculate ent H(y|x)
            计算条件信息熵
        """
        x_value_list, label_idx = np.unique(x, return_inverse=True)
        pi = np.bincount(label_idx).astype(np.float64)
        temp_ent = list()
        for x_value in x_value_list:
            sub_y = y[x == x_value]
            temp_ent.append(self.entropy(sub_y))  # 计算当一个指定的x时，y的熵
        ent = (pi / np.sum(pi)) * temp_ent
        return ent.sum()

    # 拖慢速度的罪魁祸首！！！！！！！！！！！！！！！！！！！！！！！！！！
    def two_condition_ent(self, x, y, z):
        """
        H(z|xy)
        :param x:
        :param y:
        :param z:
        :return:
        """
        HZYX = 0.0
        x_value_list = np.unique(x)
        y_value_list = np.unique(y)
        xylen = x.shape[0] * y.shape[0]
        for x_value in x_value_list:
            for y_value in y_value_list:
                sub_z = z[(x == x_value) & (y == y_value)]
                if sub_z.shape[0] != 0:
                    temp_ent = self.entropy(sub_z)  # 计算当一个指定的x时，y的熵
                    HZYX += (float(sub_z.shape[0]) / xylen) * temp_ent
        return HZYX

    def two_condition_ent1(self, condition, x):
        """
        H(z|xy)
        :param x:
        :param y:
        :param z:
        :return:
        """
        HZYX = 0.0
        d = dict()
        for i in list(range(len(s1))):
            d[s1[i]] = d.get(s1[i], []) + [x[i]]
        return sum([getEntropy(d[k]) * len(d[k]) / float(len(s1)) for k in d])

    def condition_ent_plus(self, condition, z):
        """
        多个条件的条件熵
        :param condition:里面都是一个个特征
        :param z:
        :return:
        """
        HZYX = 0.0

        condition_value = [np.unique(condition[i]) for i in range(len(condition))]  # 求出了每个子数组有哪些元素
        intersection = np.array(np.meshgrid(*condition_value)).T.reshape(-1, len(condition_value))
        number = pow(len(condition[0]), len(condition))  # 概率的分母

        for i in intersection:
            c = condition[0] == i[0]
            for t in range(1, len(i)):  # 把所有条件与一下   就是同时满足一项条件的位置了
                c = c & (condition[t] == i[t])
            sub_z = z[c]

            if sub_z.shape[0] != 0:
                temp_ent = self.entropy(sub_z)  # 计算当一个指定的x时，y的熵
                HZYX += (float(sub_z.shape[0]) / number) * temp_ent  # p(x,y)*H(z|X=x,Y=y)

        return HZYX

    # def condition_ent_thread(self, sub_z, number):
    #     HZYX = 0.0
    #     label_idx = np.unique(sub_z, return_inverse=True)[1]
    #     pi = np.bincount(label_idx).astype(np.float64)
    #     HZYX += -np.sum((pi / number) * (sub_z.shape[0] / number) * np.log(pi / number))
    #     return HZYX

    def condition_ent_plus1(self, condition, z):
        """
        多个条件的条件熵
        :param condition:里面都是一个个特征
        :param z:
        :return:
        """
        HZYX = 0.0
        # 这三行只占用很少一部分时间
        condition_value = [np.unique(condition[i]) for i in range(len(condition))]  # 求出了每个子数组有哪些元素
        intersection = np.array(np.meshgrid(*condition_value)).T.reshape(-1, len(condition_value))
        number = pow(len(condition[0]), len(condition))  # 概率的分母    联合分布有多少个值

        one = intersection[:int(intersection.shape[0] / 2)]  # 给其他的计算节点

        two = intersection[int(intersection.shape[0] / 2):]

        intersection = two

        results = []
        sp = int(intersection.shape[0] / self.num_cores)
        for i in range(self.num_cores):
            d = intersection[i * sp:(i + 1) * sp]
            tttt = [number, d, condition, z, ]
            result = self.pool.map_async(condition_ent_thread, (tttt,))
            results.append(result)
        print('apply_async: 不堵塞')

        for i in results:
            i.wait()  # 等待进程函数执行完毕

        HZYX = 0.0
        for i in results:
            if i.ready():  # 进程函数是否已经启动了
                if i.successful():  # 进程函数是否执行成功
                    HZYX += np.array(i.get())
        return HZYX

    def joint_entropy(self, x, y):
        """
        联合熵等于熵加条件熵
        :param x:
        :param y:
        :return:
        """
        HX = self.entropy(x)
        HYX = self.condition_ent(y, x)
        return HX + HYX

    def joint_entropy_three(self, x, y, z):
        """
        熵的链式法则计算
        H(x1,x2,x3)=H(x1)+H(x2|x1)+H(x3|x2x1)
        :param,x x:
        :param y:
        :param z:
        :return:
        """
        HX = self.entropy(x)
        HYX = self.condition_ent(y, x)
        HZYX = self.condition_ent_plus1([x, y], z)

        return HX + HYX + HZYX

    def joint_MI_three(self, x, y, z):
        HXYZ = self.joint_entropy_three(x, y, z)
        HX = self.entropy(x)
        HY = self.entropy(y)
        HZ = self.entropy(z)
        IXY = metrics.mutual_info_score(x, y)
        IYZ = metrics.mutual_info_score(y, z)
        IXZ = metrics.mutual_info_score(x, z)
        return HXYZ - HX - HY - HZ + IXY + IYZ + IXZ

    def cal_mi_sum(self, fi):
        """
        计算待选特征与已选中各个特征的互信息之和
        :param fi: 待选择的特征
        :param S: 已选中的特征index
        :return: 返回这些互信息的和
        """
        sum = 0.0
        for i in range(len(self.S)):
            fj = self.F[:, self.S[i]]
            MI = metrics.mutual_info_score(fi, fj)
            sum = sum + MI
        return sum

    def rmifs(self):
        row, col = self.F.shape
        F_MI = list()  # 存的每个特征和标签的互信息  对应的特征也都是没有被选进S的
        F_MI_index = list()  # 存的原始下标

        # 把每个特征与标签的互信息都存到了F_MI
        for i in range(col):
            fi = self.F[:, i]
            MI = metrics.mutual_info_score(fi, self.C)
            F_MI.append(MI)
            F_MI_index.append(i)

        # 互信息最大的作为第一个特征
        maxindex = max(F_MI)
        i = F_MI.index(maxindex)
        self.S.append(F_MI_index[i])  # S里存的是原集合的下标
        F_MI.pop(i)
        F_MI_index.pop(i)
        self.m = self.m - 1  # 表示已经找到一个特征了

        if self.m != 0:
            MI_fi_S = list()  # 这个相当于一轮的MI  因为S会变  待选和S的互信息就得重新求了    这个里面最大的要添加进S
            for t1 in range(len(F_MI)):
                # 这个传入的F里不应包括fi
                # 但是如果删掉  就无法对应他是第几个了   所以需要两个列表 index，他与标签的互信息
                fi = self.F[:, F_MI_index[t1]]
                sum = self.cal_mi_sum(fi)
                difference = F_MI[t1] - sum * self.beta
                MI_fi_S.append(difference)

            maxindex = MI_fi_S.index(max(MI_fi_S))  # 找到这些和里面最大的那个
            self.S.append(F_MI_index[maxindex])  # 把特征的原始索引加到已选特征里
            F_MI.pop(maxindex)
            F_MI_index.pop(maxindex)
            self.m = self.m - 1

        # 到这里已经使用mifs挑选出两个特征来了   把他们的原始下标存在了self.S里

        mi_sum = list()
        for times in range(self.m):
            for t1 in range(len(F_MI)):
                # 这个传入的F里不应包括fi
                # 但是如果删掉  就无法对应他是第几个了   所以需要两个列表 index，他与标签的互信息
                fi = self.F[:, F_MI_index[t1]]
                #############################################
                sum_betabehind = 0.0
                for j in range(len(self.S)):
                    fj = self.F[:, self.S[j]]
                    for k in range(j + 1, len(self.S)):
                        fk = self.F[:, self.S[k]]
                        MI_fi_fj = metrics.mutual_info_score(fi, fj)
                        MI_fi_fk = metrics.mutual_info_score(fi, fk)
                        MI_fi_fj_fk = self.joint_MI_three(fi, fj, fk)
                        sum_betabehind += MI_fi_fj + MI_fi_fk - MI_fi_fj_fk
                ###########################################
                difference = F_MI[t1] - sum_betabehind * self.beta
                mi_sum.append(difference)

            maxindex = mi_sum.index(max(mi_sum))  # 找到这些和里面最大的那个
            self.S.append(F_MI_index[maxindex])  # 把特征的原始索引加到已选特征里
            F_MI.pop(maxindex)
            F_MI_index.pop(maxindex)

        print(self.S)
        return self.S

    def rmifs2classifer(self):
        return self.F[:, self.S], self.C, self.test_X[:, self.S], self.test_y

    def rmifs_thread(self):
        self.rmifs()

        return self.rmifs2classifer()


if __name__ == '__main__':
    data = np.array([[0, 1963, 0],
                     [0, 363, 0],
                     [0, 425, 0],
                     [0, 936, 2],
                     [1265, 256, 4755],
                     [0, 95, 0],
                     [98, 36, 1374],
                     [0, 1751, 0]])
    totalval = float(np.sum(data))
    data = (data) / totalval  # 求联合概率分布
    # print(data)
    print(calcConditionalEnt(data))
