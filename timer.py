# -*- coding: utf-8 -*-

# ***************************************************
# * File        : timer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-22
# * Version     : 0.1.052220
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import datetime as dt
import time
import functools

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Timer():
    """
    运行时间计算
    """
    
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        logger.info(f"Time taken: {end_dt - self.start_dt}")


def timeit_func1(func):
    """
    分析代码运行时间
    """
    @functools.wraps(func)
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        print(f"function used: {end - start} second.")
    
    return wrapper


def timeit_func2(func):
    """
    分析代码运行时间
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"function used: {end - start} seconds.")
    
    return wrapper




# 测试代码 main 函数
def main():
    @timeit_func1
    def test_func():
        print("a")

    test_func()

if __name__ == "__main__":
    main()
