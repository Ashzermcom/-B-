import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

data = pd.read_excel("/Users/ashzerm/item/GasOline/data/oline.xlsx")
target = np.array(data['RON_LOSS'].copy())
data = data[data.columns[16:]]
data = np.array(data)

estimator = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=5)

estimator.fit()
selector = RFE(estimator=estimator, n_features_to_select=10)
selector.fit(data, target)

print("N_features {}".format(selector.n_features_))
print("Support is {}".format(selector.support_))
print("Ranking is {}".format(selector.ranking_))
