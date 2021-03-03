# coding:utf-8
# MI(X1;X2|Y)=H(X1，Y)+H(X2，Y)-H(Y)-H(X1，X2，Y)
import numpy as np
from sklearn import metrics
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from util.mutualInfor import MI_chain, condition_ent, condition_ent_plus_multiprocess


class cmi:
    def __init__(self, F, C, m=3):
        self.F = F
        self.C = C  # 标签
        self.m = m  # 要选取m个特征  最小为1
        self.selectedFeature = list()
        self.num_cores = 4

        self.PoolType = 'Process'
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            # self.num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(self.num_cores)  # 设置池的大小  为计算机核心数

    def cmi(self):
        S = list()
        row, col = self.F.shape
        F_MI = list()
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

        for times in range(self.m):
            CMI = list()
            for t1 in range(len(F_MI)):
                print('1')
                # 这个传入的F里不应包括fi
                # 但是如果删掉  就无法对应他是第几个了   所以需要两个列表 index，他与标签的互信息
                fi = self.F[:, F_MI_index[t1]]
                IfiC = F_MI[t1]  # 把对应位置的fi和标签的互信息取出来
                IfiS = MI_chain(fi, self.F, S)
                cond = self.F[:, S]
                cond = cond.reshape(cond.shape[1], -1)
                c = np.row_stack((cond, self.C))
                IfiSC = condition_ent(self.C, fi) - condition_ent_plus_multiprocess(c, fi, self.pool, self.num_cores)
                c = IfiC - IfiS + IfiSC
                CMI.append(c)

            maxindex = CMI.index(max(CMI))  # 找到这些和里面最大的那个
            S.append(F_MI_index[maxindex])  # 把特征的原始索引加到已选特征里
            F_MI.pop(maxindex)
            F_MI_index.pop(maxindex)
        return S

    def cmi2classifer(self):
        return self.F[:, self.selectedFeature], self.C

    def cmi_thread(self):
        self.selectedFeature = self.cmi()
        return self.cmi2classifer()


if __name__ == '__main__':
    pass
