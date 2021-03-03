# coding:utf-8

from sklearn import metrics


class MIFS:
    """
    具有m个特征的特征子集S
    这些只需要处理训练集  测试集当然需要处理了   你筛选出特征来测试集那么多特征怎么用   F是训练集  原始特征

    """

    def __init__(self, F, C, m=3, beta=0.5):
        self.F = F
        self.C = C  # 标签
        self.selectedFeature = list()  # 已选特征的下标
        self.m = m  # 要选取m个特征  最小为1
        self.beta = beta

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

    def mifs(self):
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

        # 写错了   不是和已选特征的互信息和最大    要改成互信息减去这个和还最大的那个  已经改好了
        for times in range(self.m):
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
        return S

    def mifs2classifer(self):
        return self.F[:, self.selectedFeature], self.C

    def mifs_thread(self):
        self.selectedFeature = self.mifs()  # 如果只需要特征索引就只用这个
        return self.mifs2classifer()
