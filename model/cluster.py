import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


class ClusterModel:
    def __init__(self, num_clusters):
        self.cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean')
        self.labels = None
        self.core_nums = []

    def train(self, x, method='Q'):
        """
        'Q' 型聚类为对样本进行聚类
        'R' 型聚类为堆指标进行聚类
        Args:
            x: np.array
            method: str
        return:
        """
        if method not in ['Q', 'R']:
            print("当前{}方法暂时不支持".format(method))
            return
        if method == 'Q':
            self.labels = self.cluster.fit_predict(x)
        elif method == 'R':
            self.labels = self.cluster.fit_predict(x.T)
        return self.labels


if __name__ == '__main__':
    model = ClusterModel(num_clusters=15)
    import numpy as np
    import pandas as pd
    data = pd.read_excel("./data/oline.xlsx")
    data = data[data.columns[16:]]
    data = np.array(data)
    res = model.train(data, method='R')
    print(res)

