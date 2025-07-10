# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ()
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : yyyy-mm-dd
# * Version     : 0.1.0
# * Description : 评估函数运行时间
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
import time
import functools

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


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


def timeit_script():
    pass




# 测试代码 main 函数
def main():
    @timeit_func1
    def test_func():
        print("a")

    test_func()

if __name__ == "__main__":
    main()
