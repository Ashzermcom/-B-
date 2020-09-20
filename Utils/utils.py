import numpy as np

def nan_data_rate(df, n, ascending_=False, origin=True):
    """
    【Function】缺失率统计函数 nan_data_rate
    :param df: 需要处理的数据框
    :param n: 显示变量个数
    :param ascending_: 按缺失程度上升还是下降表示
    :param origin: 是否显示无缺失值失变量
    :return: 返回前n个缺失变量缺失率
    """
    if n > len(df.columns):  # 判断显示个数是否多于变量总数,如果超过则默认显示全部变量
        print('显示变量个数多于变量总数%i,将显示全部变量' % (len(df.columns)))
        n = len(df.columns)
    na_rate = df.isnull().sum() / len(df) * 100  # 统计各变量缺失率
    if origin:  # 判断为真则显示无缺失值的变量
        na_rate = na_rate.sort_values(ascending=ascending_)
        missing_data = pd.DataFrame({'Missing_Ratio': na_rate})
    else:  # 判断为负则显示只有缺失值的变量
        na_rate = na_rate.drop(na_rate[na_rate == 0].index).sort_values(ascending=ascending_)
        missing_data = pd.DataFrame({'Missing_Ratio': na_rate})
    return missing_data.head(n)


# 最小最大标准化
def min_max_normalize(df):
    """
        将数据框中所有变量 最小最大值标准化
    Args:
        df: 原始未标准化的数据框
    return: 标准化后的数据框
    """
    #
    for col in df.columns:
        min_val = np.min(df[col])
        max_val = np.max(df[col])
        df[col] = (df[col] - min_val) / (max_val - min_val)
        print("{} 变量已经被最小最大标准化".format(col))
    return df