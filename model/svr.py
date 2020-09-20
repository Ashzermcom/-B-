import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SVRModel:
    def __init__(self, is_intercept=False):
        self.svr = LinearSVR(fit_intercept=is_intercept)
        self.score = None
        self.mae = None
        self.mse = None

    def train(self, x, y):
        self.svr.fit(x, y)
        self.score = self.svr.score(x, y)
        print('SVR Coefficients: {}'.format(self.svr.coef_))
        print('SVR Score: {}'.format(self.score))

    def test(self, x, y):
        y_pre = self.svr.predict(x)
        self.mae = mean_absolute_error(y_pre, y)
        self.mse = mean_squared_error(y_pre, y)
        print("SVR的预测mae是{}, mse是{}".format(self.mae, self.mse))


if __name__ == '__main__':
    model = SVRModel(is_intercept=False)
    import numpy as np
    import pandas as pd
    data = pd.read_excel("/Users/ashzerm/item/GasOline/data/oline.xlsx")
    target = np.array(data['RON_LOSS'].copy())
    data = data[data.columns[16:]]
    data = np.array(data)
    model.train(data, target)
    model.test(data, target)

