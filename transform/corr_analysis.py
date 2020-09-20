import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Utils.utils import min_max_normalize

corr_val = ['D121去稳定塔流量', '还原器温度',
            'D-123压力', 'D-121水液位', 'D-102温度', '原料汽油硫含量', '稳定塔顶回流流量',
            '空气预热器空气出口温度', '低压热氮气压力', 'R-101下部床层压降', 'R-101床层下部温度',
            'P-101B入口过滤器差压', 'ME-105过滤器压差', 'F-101辐射室出口压力',
            'S_ZORB AT-0011', 'D-201含硫污水液位', 'D101原料缓冲罐压力']


# 变量间相关性分析
def corr_plot(df, columns=corr_val):
    data_cor = min_max_normalize(df[columns])
    corrmat = data_cor.corr()
    # 设置下三角样式
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # 创建调试盘
    cmap = sns.diverging_palette(0.5, 10, as_cmap=True)
    # 绘制热力图
    plt.figure(figsize=(16, 9))
    sns.heatmap(corrmat, mask=mask, cmap=cmap, annot=True, linewidths=0.2,
                center=100, vmax=1, vmin=0, square=True)
    plt.show()
    return


if __name__ == '__main__':
    test = pd.read_excel('/Users/ashzerm/item/GasOline/data/oline.xlsx')
    corr_plot(test)
