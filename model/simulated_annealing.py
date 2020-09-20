import random
import numpy as np


class SimulatedAnnealing:
    def __init__(self, x, delta, max_limit, min_limit):
        """
        Args:
            x (np.array()): 操作变量, size(1,p), p为操作变量个数
            delta (np.array()): 每个操作变量每次可以调整的值, size(1,p), p为操作变量数
            max_limit (np.array()): 每个操作变量取值范围的上限
            min_limit (np.array()): 每个操作变量取值范围的下限
        """
        self.x = x
        self.delta = delta
        self.max_limit = max_limit
        self.min_limit = min_limit

    def update(self, x, index, flag):
        """
        更新参数
        Args:
            x (np.array()): 操作变量, size(1,p), p为操作变量个数
            index: 需要更新的参数的索引
            flag: 需要更新的参数的操作符号
        """
        for i in range(len(index)):
            if flag[index[i]]:
                # 当前符号为+时, True
                temp = x[index[i]] + self.delta[index[i]]
                if temp > self.max_limit[index[i]]:
                    # 当更新后越过上界时,更改符号为-号False,不更新参数
                    flag[index[i]] = False
                else:
                    # 否则更新参数
                    x[index[i]] = temp
            else:
                # 当前符号位-时, False
                temp = x[index[i]] - self.delta[index[i]]
                if temp < self.min_limit[index[i]]:
                    # 当更新后越过下界时,更改符号为+号True
                    flag[index[i]] = True
                else:
                    # 否则更新参数
                    x[index[i]] = temp

    def reverse(self, x, index, flag):
        """
        复原参数
        Args:
            x (np.array()): 操作变量, size(1,p), p为操作变量个数
            index: 需要更新的参数的索引
            flag: 需要更新的参数的操作符号
        """
        for i in range(len(index)):
            if flag[index[i]]:
                x[index[i]] -= self.delta[index[i]]
            else:
                x[index[i]] += self.delta[index[i]]
            flag[index[i]] = bool(1 - flag[index[i]])  # 符号取反

    def train(self, x, func, temperature: int, num_value_per_iter: int, max_iter: int, target: int, alpha=0.999):
        """
        Args:
            x (np.array()): 操作变量, size(1,p), p为操作变量个数
            func (model): 用于计算变量目标值的函数
            temperature (int): 模拟退火温度值
            max_iter (int): 最大迭代次数
            num_value_per_iter (int): 每轮搜索的变量个数
            target (int): 初始目标值
            alpha: temperature的衰减系数
        """
        # ----- 判断输入异常 ----- #
        if len(x.shape) > 1:
            raise ValueError("输入x的size必须是 1xp")
        num = x.shape[0]
        if np.sum(x < self.max_limit) != num or np.sum(x > self.min_limit) != num:
            raise ValueError("当前变量x不是可行解")

        # 选取变换变量
        flag = np.array([False] * 2)  # 初始化操作变量 用True表示+ False表示-
        index = np.random.randint(0, num, 2)
        target_ori = target
        t = temperature
        # ----- 更新参数 ----- #
        for _ in range(max_iter):
            # 更新操作参数
            self.update(x, index, flag)
            target_pre = func.predict(x)
            # 当比当前目标值小时退出循环
            if target_pre <= target_ori:
                continue
            else:
                # 计算接受概率
                p_accept = np.exp((target_pre - target_ori) / (t + 1e-10))
                if random.uniform(0, 1) < p_accept:
                    continue
                else:
                    # 复原已经更新的参数并将更新符号取反
                    self.reverse(x, index, flag)
            t *= alpha
        return x
