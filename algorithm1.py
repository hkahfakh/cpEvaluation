# coding:utf-8
import numpy as np
from itertools import combinations
from sklearn import metrics
from util.mutualInfor import condition_MI


class Banzhaf:
    def __init__(self, F, C, Omega=3, Tau=0.5):
        self.F = F
        self.C = C
        self.Tau = Tau
        self.Omega = Omega

    def __del__(self):
        print('Banzhaf对象被回收---')

    # Input: original feature sub feature fi F, limit value Omega
    # Output: coalition array
    # just use index as F not real data
    # OK
    def creatCoalitionis(self):
        list2 = []
        for i in range(1, self.Omega + 1):
            iter = combinations(range(self.F.shape[1]), i)
            list2.extend(list(iter))
        return (list2)

    # Input: 传入特征下标
    # Output: coalition array
    # 所有的都传下标吧
    def relationshiDetection(self, j, i):
        fi = self.F[:, i]
        fj = self.F[:, j]
        MI1 = condition_MI(fj, self.C, fi)
        MI2 = metrics.mutual_info_score(fj, self.C)
        return True if MI1 <= MI2 else False

    def getFeatureNumber(self, coalition, i):
        zi, mi = 0, 0
        for j in coalition:
            if self.relationshiDetection(j, i):
                zi = zi + 1  # 相互依赖
            else:
                mi = mi + 1  # 冗余或独立
        return zi, mi

    # Input: coalition , threshold Tau
    # Output: Marginal contribution
    # 这是边际贡献  也是并上特征fi后联盟的胜败
    def calcPayoff(self, coalition, fi):
        zi, mi = self.getFeatureNumber(coalition, fi)
        # Zi（K）是与特征fi相互依赖的特征个数
        # mi（K）是与特征fi冗余或独立的特征个数。
        if mi <= 0:
            return 0
        p = zi / mi
        return 1 if p > self.Tau else 0

    # Input: 边际贡献
    # Output: Banzhaf power index
    def calcBanzhaf(self, payoff):
        payoff = np.array(payoff)
        col = np.size(payoff)
        sum = np.sum(payoff)
        return sum / (2 ^ (col - 1))

    # Input:
    # Output: normalized Pv vector
    def normalizedPv(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    # Input: A training sample O with feature space F and the target C.
    # Output: Pv: Banzhaf power index vector of F.
    def banzhafPowerIndex(self):
        Pv = []
        for i in range(self.F.shape[1]):
            payoff = []
            coalitions = self.creatCoalitionis()
            for coalition in coalitions:
                payoff.append(self.calcPayoff(coalition, i))
            bi = self.calcBanzhaf(payoff)
            Pv.append(bi)
        return np.array(Pv)
        # return self.normalizedPv(Pv)

    def updataF(self, F, C):
        self.F = F
        self.C = C
