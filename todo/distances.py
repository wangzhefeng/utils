# -*- coding: utf-8 -*-

# ***************************************************
# * File        : distance.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101716
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

import fastdtw
from dtaidistance import dtw
from dtaidistance import dtw_vis

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class distance:

	def _init__(self, method):
		self.method = method

	def Eula(self):
		pass


class TimeseriesDistance(object):
	
    def __init__(self):
          pass

    def DTW(self):
        pass






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
