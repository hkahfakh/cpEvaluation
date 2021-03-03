# coding:utf-8
from util import getData as gd
from algorithm1 import Banzhaf
from algorithm2 import featureSelection
from featuresAlgorithm.mifs import MIFS

if __name__ == '__main__':
    data = gd.get_data("./dataSet/glass.npy")
    X, y = data[:, :-1], data[:, -1]
    # 拆分训练集和测试集 0.7
    train_X, train_y, test_X, test_y = gd.data_split(X, y, rate=0.9)

    m = MIFS(train_X, train_y)
    t = m.mifs()

    b = Banzhaf(train_X, train_y)
    # Pv = b.banzhafPowerIndex()
    sf = featureSelection(train_X, train_y, 3)  # sf是选好的特征
