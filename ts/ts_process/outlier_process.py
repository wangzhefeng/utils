# -*- coding: utf-8 -*-

# ***************************************************
# * File        : outlier_process.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-24
# * Version     : 1.0.052422
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def outlier_replace_with_ma(df: pd.DataFrame, col_with_outlier: str):
    """
    用移动平均值填充异常值

    Args:
        df (pd.DataFrame): 待处理数据集
        col_with_outlier (str): 需要进行异常值处理的变量名

    Returns:
        _type_: _description_
    """
    df["moving_avg"] = df[col_with_outlier].rolling(window=5, min_periods=1).mean()
    # 计算 z-score
    df["z-score"] = (df[col_with_outlier] - df[col_with_outlier].mean()) / df[col_with_outlier].std()
    # 将异常值替换为移动平均值
    df.loc[df["z-score"].abs() > 1.5, col_with_outlier] = df["moving_avg"]
    del df["moving_avg"]
    del df["z-score"]

    return df




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
