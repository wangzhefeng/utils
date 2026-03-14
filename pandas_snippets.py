# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pandas_snippets.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-01-17
# * Version     : 1.0.011717
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
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, date

import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ##############################
# pandas DataFrame style 
# ##############################
df = pd.read_csv("../input/acea-water-prediction/Aquifer_Petrignano.csv")
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
df.head().style.set_properties(subset=["date"], **{"background-color": "lightblue"})




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
