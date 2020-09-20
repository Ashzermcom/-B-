import numpy as np
import pandas as pd

df = pd.read_excel('../data/stand_oline.xlsx')
# print(df.loc[(df['RON_LOSS'] > 1.5)].index)


def sigma3(data, columns=['RON_LOSS']):
    for col in columns:
        mu = np.mean(data[col])
        sigma = np.std(data[col])
        max_limit = mu + 3 * sigma
        min_limit = mu - 3 * sigma
        data.drop(index=data.loc[(data[col] > max_limit)].index, inplace=True)
        data.drop(index=data.loc[(data[col] < min_limit)].index, inplace=True)
        print(data.loc[(data[col] > max_limit)].index)
        print(data.loc[(data[col] < min_limit)].index)
    return data


result = sigma3(df)
result.to_excel('../data/result.xlsx')
