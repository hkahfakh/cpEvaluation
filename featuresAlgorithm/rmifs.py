# coding:utf-8
from sklearn import metrics
import numpy as np
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from util.mutualInfor import joint_entropy_three, entropy, joint_MI_three


class Rmifs:
    def __init__(self, F, C, m=3, beta=0.5):
        self.F = F
        self.C = C  # 标签
        self.selectedFeature = list()  # 已选特征的下标
        self.m = m  # 要选取m个特征  最小为1
        self.beta = beta  # 论文上说rmifs的β与mifs的是倍数关系
        self.num_cores = 4

        self.PoolType = 'Process'
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            # self.num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(self.num_cores)  # 设置池的大小  为计算机核心数

    def cal_mi_sum(self, fi):
        """
        计算待选特征与已选中各个特征的互信息之和
        :param fi: 待选择的特征
        :param S: 已选中的特征index
        :return: 返回这些互信息的和
        """
        sum = 0.0
        for i in range(len(self.selectedFeature)):
            fj = self.F[:, self.selectedFeature[i]]
            MI = metrics.mutual_info_score(fi, fj)
            sum = sum + MI
        return sum

    def rmifs(self):
        S = list()
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
        S.append(F_MI_index[i])  # S里存的是原集合的下标
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
            S.append(F_MI_index[maxindex])  # 把特征的原始索引加到已选特征里
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
                for j in range(len(S)):
                    fj = self.F[:, S[j]]
                    for k in range(j + 1, len(S)):
                        fk = self.F[:, S[k]]
                        MI_fi_fj = metrics.mutual_info_score(fi, fj)
                        MI_fi_fk = metrics.mutual_info_score(fi, fk)
                        MI_fi_fj_fk = joint_MI_three(fi, fj, fk)
                        sum_betabehind += MI_fi_fj + MI_fi_fk - MI_fi_fj_fk
                ###########################################
                difference = F_MI[t1] - sum_betabehind * self.beta
                mi_sum.append(difference)

            maxindex = mi_sum.index(max(mi_sum))  # 找到这些和里面最大的那个
            S.append(F_MI_index[maxindex])  # 把特征的原始索引加到已选特征里
            F_MI.pop(maxindex)
            F_MI_index.pop(maxindex)
        return S

    def rmifs2classifer(self):
        return self.F[:, self.selectedFeature], self.C

    def rmifs_thread(self):
        self.selectedFeature = self.rmifs()
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
    # print(calcConditionalEnt(data))
