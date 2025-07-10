# -*- coding: utf-8 -*-

# ***************************************************
# * File        : collections_demo.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-10
# * Version     : 1.0.071017
# * Description : collections 库
# *                1.namedtuple()
# *                2.deque
# *                3.ChainMap
# *                4.Counter
# *                5.OrderedDict
# *                6.defaultdict
# *                7.UserDict
# *                8.UserList
# *                9.UserString
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger







# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
