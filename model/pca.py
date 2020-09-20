import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA


def disperse(df, col='RON_LOSS'):
    bins = [-0.1, 0.6, 1.0, 1.5, 2, 3]
    x = pd.cut(df[col], bins=bins, labels=['0.6以下', '0.6-1.0', '1.0-1.5', '1.5-2', '2-3'])
    return x


class PCAModel:
    def __init__(self, num_components):
        self.num_components = num_components
        self.pca = PCA(n_components=num_components)
        self.explained_variance = None
        self.explained_ratio = None

    def train(self, x, is_show=False):
        """
            训练主成分分析模型并执行降维
        :param x:
        :param is_show:
        :return:
        """
        self.pca.fit(x)
        x_pca = self.pca.transform(x)
        self.explained_ratio = self.pca.explained_variance_ratio_
        self.explained_variance = self.pca.explained_variance_
        print("当前的主成分方差为: {}".format(self.explained_variance))
        print("当前的主成分方差占比为: {}".format(self.explained_variance))
        if is_show:
            self._plot2d_pca(x, (0, 1))
        return x_pca

    @staticmethod
    def _plot2d_pca(x, idx):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x[:, idx[0]], x[:, idx[1]], color='red')
        ax.set_xlabel("x[{}]列".format(idx[0]))
        ax.set_ylabel("x[{}]列".format(idx[1]))
        # ax.legend(loc="best")
        ax.set_title("石油操作主成分")
        plt.show()


class KernelPCAModel:
    def __init__(self, num_components, kernel):
        self.num_components = num_components
        self.kernel = kernel
        self.kernel_pca = KernelPCA(n_components=num_components, kernel=kernel)
        self.explained_variance = None
        self.explained_ratio = None

    def test_pca_kernel(self, x):
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for ker in kernels:
            k_pca = KernelPCA(n_components=self.num_components, kernel=ker)
            k_pca.fit(x)
            print('Kernel = {} --> lambdas: '.format(ker))
            print(k_pca.lambdas_)

    def plot_KPCA_rbf(self, x, y):
        fig = plt.figure(figsize=(12, 10))
        colors = [(1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0,3,0.7,0),(0.7,0.3,0)]
        gammas = [0.5, 1, 4, 10]
        for i, gamma in enumerate(gammas):
            k_pca = KernelPCA(n_components=self.num_components, kernel='rbf', gamma=gamma)
            k_pca.fit(x)
            x_pca = k_pca.transform(x)
            ax = fig.add_subplot(2, 2, i + 1)
            for lab, color in zip(np.unique(y), colors):
                position = y == lab
                ax.scatter(x_pca[position, 0], x_pca[position, 1], label=lab, color=color)
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.legend(loc='best')
            ax.set_title(r"$\exp(-{}||x - z||^2)$".format(gamma))
        plt.suptitle("KPCA-rbf")
        plt.show()

    def train(self, x):
        self.kernel_pca.fit(x)
        return self.kernel_pca.transform(x)


if __name__ == '__main__':
    # model = PCAModel(num_components=30)
    model = KernelPCAModel(num_components=20, kernel='rbf')
    import numpy as np
    import pandas as pd

    data = pd.read_excel("../data/stand_oline.xlsx")
    target = disperse(data, 'RON_LOSS')
    data = data[data.columns[10:]]
    print(data.columns)
    data = np.array(data)
    model.test_pca_kernel(data)
    model.plot_KPCA_rbf(data, target)
    # print("方差解释总比例为{}".format(sum(model.explained_ratio)))
    # print(res)
