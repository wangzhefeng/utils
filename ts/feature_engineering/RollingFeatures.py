# -*- coding: utf-8 -*-

# ***************************************************
# * File        : RollingFeatures.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042415
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class RollingFeatures:
    
    def __init__(self, data, window_length: int = 7) -> None:
        self.data = data
        self.window_length = window_length

    def features(self, raw_feature, new_feature):
        # 滚动窗口特征
        self.data[new_feature] = self.data[raw_feature].rolling(self.window_length).mean()
        # 重命名
        data_columns = ["ts", raw_feature, new_feature]
        
        self.data = self.data[data_columns]




# 测试代码 main 函数
def main():
    # 数据读取
    series = pd.read_csv(
        "E:/projects/timeseries_forecasting/tsproj/dataset/daily-minimum-temperatures-in-me.csv",
        header = 0,
        index_col = 0,
        # parse_dates = [0],
        # date_parser = lambda dates: pd.to_datetime("190" + dates, format = "%Y-%m"),
    )
    print(series.head())
    
    temps = pd.DataFrame(series.values)
    print(temps.head())

    width = 2
    shifted = temps.shift(width - 1)
    print(shifted)
    
    # 向上统计的步长
    window = shifted.rolling(window = 4)
    # 分别取过去 4 天的最小值、均值、最大值
    df = pd.concat([
        window.min(),
        window.mean(),
        window.max(),
        temps,
    ], axis = 1)
    df.columns = ["min", "mean", "max", "t+1"]
    print(df.head())

if __name__ == "__main__":
    main()
