from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


def xgb_parameters_search(x, y):
    xgb_r = XGBRegressor()
    xgb_parameters = {
        'max_depth': [5, 7, 10],
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05],
    }
    xgb_gs = GridSearchCV(estimator=xgb_r, param_grid=xgb_parameters, cv=5)
    xgb_gs.fit(x, y)
    print('xgb 的最佳模型得分为: {}'.format(xgb_gs.best_score_))
    print('xgb 的最优参数组合为: ', xgb_gs.best_params_)
    return xgb_gs.best_params_
