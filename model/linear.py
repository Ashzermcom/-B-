import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


class LassoModel:
    def __init__(self, alpha):
        self.model = Lasso(alpha=alpha, selection='random')
        self.score = None
        self.mae = None
        self.mse = None

    def train(self, x, y, num_folds):
        folder = KFold(n_splits=num_folds, random_state=2, shuffle=True)
        for i, train_idx, test_idx in enumerate(folder.split(x, y)):
            x_train, y_train = x[train_idx, :], y[train_idx, :]
            x_test, y_test = x[test_idx, :], y[test_idx, :]
            self.model.fit(x_train, y_train)
            self.score = self.model.score(x_train, y_train)
            y_pre = self.model.predict(x_test)
            mae = mean_absolute_error(y_pre, y_test)
            mse = mean_squared_error(y_pre, y_test)
            print("Lasso's {} fold Coefficients: {}".format(i, self.model.coef_))
            print("Lasso's {} fold Score: {}".format(i, self.score))
            print("Lasso's {} fold mean absolute error: {}".format(i, mae))
            print("Lasso's {} fold mean squared error: {}".format(i, mse))

    def test(self, x, y):
        y_pre = self.model.predict(x)
        self.mae = mean_absolute_error(y_pre, y)
        self.mse = mean_squared_error(y_pre, y)
        print("Lasso的预测mae是{}, mse是{}".format(self.mae, self.mse))

    @staticmethod
    def test_lasso_alpha(x, y, alphas=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=2)
        scores = []
        for i, alpha in enumerate(alphas):
            model = Lasso(alpha=alpha)
            model.fit(x_train, y_train)
            scores.append(model.score(x_test, y_test))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, scores)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"score")
        ax.set_xscale('log')
        ax.set_title('Lasso')
        plt.show()


class LinearModel:
    def __init__(self, alpha):
        self.model = LinearRegression(fit_intercept=False)
        self.score = None
        self.mae = None
        self.mse = None

    def train(self, x, y):
        self.model.fit(x, y)
        self.score = self.model.score(x, y)
        print('Linear Coefficients: {}'.format(self.model.coef_))
        print('Linear Score: {}'.format(self.score))

    def test(self, x, y):
        y_pre = self.model.predict(x)
        self.mae = mean_absolute_error(y_pre, y)
        self.mse = mean_squared_error(y_pre, y)
        print("Linear的预测mae是{}, mse是{}".format(self.mae, self.mse))


class ElasticNetModel:
    def __init__(self):
        self.model = ElasticNet(selection='random')
        self.score = None
        self.mae = None
        self.mse = None

    def train(self, x, y):
        self.model.fit(x, y)
        self.score = self.model.score(x, y)
        print('SVR Coefficients: {}'.format(self.model.coef_))
        print('SVR Score: {}'.format(self.score))

    def test(self, x, y):
        y_pre = self.model.predict(x)
        self.mae = mean_absolute_error(y_pre, y)
        self.mse = mean_squared_error(y_pre, y)
        print("SVR的预测mae是{}, mse是{}".format(self.mae, self.mse))

    @staticmethod
    def grid_search_alpha_rho(x, y):
        """
            通过GridSearch搜索最优模型参数
        Args:
            x: 训练集自变量
            y: 训练集因变量
        return: 最优参数组合
        """
        els_parameters = {
            'alpha': np.logspace(-2, 2),
            'l1_ratio': np.linspace(0.01, 1)
        }
        eln = ElasticNet()
        eln_gs = GridSearchCV(estimator=eln, param_grid=els_parameters, cv=5)
        eln_gs.fit(x, y)
        print('Elastic 最大回归系数为 {}'.format(eln_gs.best_score_))
        return eln_gs.best_params_

    @staticmethod
    def test_elastic_net_alpha_rho(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=2)
        alphas = np.logspace(-2, 2)
        rhos = np.linspace(0.01, 1)
        scores = []
        for alpha in alphas:
            for rho in rhos:
                model = ElasticNet(alpha=alpha,l1_ratio=rho)
                model.fit(x_train, y_train)
                scores.append(model.score(x_test, y_test))
        # 绘制三维图
        alphas, rhos = np.meshgrid(alphas, rhos)
        scores = np.array(scores).reshape(alphas.shape)
        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\rho$")
        ax.set_zlabel('score')
        ax.set_title('ElasticNet')
        plt.show()


if __name__ == '__main__':
    line_model = LassoModel(alpha=0.01)
    import numpy as np
    import pandas as pd

    data = pd.read_excel("/Users/ashzerm/item/GasOline/data/oline.xlsx")
    target = np.array(data['RON_LOSS'].copy())
    value = data[data.columns[16:]].copy()
    value = np.array(value)
    line_model.test_lasso_alpha(value, target)
    # model.train(data, target)
    # model.test(data, target)
    els = ElasticNetModel(0.01)
    els.test_elastic_net_alpha_rho(data, target)
