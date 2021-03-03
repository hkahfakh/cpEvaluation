# coding:utf-8
from sklearn import metrics
import numpy as np

class mim:
    def __init__(self, F, C, featureNumber=5):
        self.F = F  # 特征集合
        self.C = C  # 类标签
        self.selectedFeature = list()  # 存储选出来的特征在F中的下标
        self.featureNumber = featureNumber

    def calcMI(self):
        # 把互信息都存到了S_MI
        row, col = self.F.shape
        S_MI = list()
        for i in range(col):
            fi = self.F[:, i]
            MI = metrics.mutual_info_score(fi, self.C)
            S_MI.append(MI)
        return np.array(S_MI)

    def selectFeature(self, S_MI):
        S = list()
        for j in range(self.featureNumber):
            m = max(S_MI)
            S.append(S_MI.index(m))
            S_MI[S_MI.index(m)] = -1
        return S

    def mim(self):
        '''
        选出与类标签之间互信息最大的几个特征
        :param F: 原始特征集合
        :param C: 标签
        :param featureNumber:
        :return: S 已选特征集合
        '''
        S_MI = self.calcMI()
        S = self.selectFeature(S_MI)
        return S

    def mim2classifer(self):
        """
        把选好的特征取出来
        给到分类器进行分类操作
        :return: 根据下标返回实际的特征内容
        """
        return self.F[:, self.selectedFeature], self.C

    def mim_thread(self):
        """
        如果要给分类器  运行这个就行了
        :return:
        """
        self.selectedFeature = self.mim()
        return self.mim2classifer()


if __name__ == '__main__':
    mim()
