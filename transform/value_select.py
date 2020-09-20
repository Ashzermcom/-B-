import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def feature_select():
    df = pd.read_excel("/Users/ashzerm/item/GasOline/data/stand_oline.xlsx")
    target = np.array(df['RON_LOSS'].copy())
    df.drop('RON_LOSS', axis=1, inplace=True)
    df.drop('产品辛烷值RON', axis=1, inplace=True)
    df.drop('产品硫含量', axis=1, inplace=True)
    data = np.array(df)
    estimator = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=5)
    selector = RFE(estimator=estimator, n_features_to_select=25)
    selector.fit(data, target)

    print("N_features {}".format(selector.n_features_))
    print("Support is {}".format(selector.support_))
    print("Ranking is {}".format(selector.ranking_))
    print("选取的特征为: ", df.columns[selector.ranking_ == 1])


result = ['D121去稳定塔流量', '还原器温度', 'E-101D壳程出口管温度', 'D-204液位', 'D123冷凝水罐液位',
          'D-123压力', 'D-121水液位', 'D-102温度', '原料汽油硫含量',
          'TAG表和PID图未见PDI-2107点，是否为DI-2107', '稳定塔顶回流流量', '热循环气去R101底提升气管流量',
          '空气预热器空气出口温度', '低压热氮气压力', 'R-101下部床层压降', 'R-101床层中部温度', 'R-101床层下部温度',
          'P-101B入口过滤器差压', 'ME-109过滤器差压', 'ME-105过滤器压差', 'F-101辐射室出口压力',
          'S_ZORB AT-0004', 'S_ZORB AT-0011', 'D-201含硫污水液位', 'D101原料缓冲罐压力']

if __name__ == '__main__':
    feature_select()
