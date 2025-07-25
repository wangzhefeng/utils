# -*- coding: utf-8 -*-

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# **********************************************

# python libraries
from pipe import select, where


# 测试代码 main 函数
def main():
    # map filter
    arr = [1, 2, 3, 4, 5]
    print(list(
        map(
            lambda x: x * 2, 
            filter(lambda x: x % 2 == 0, arr)
        )
    ))

    # pipe
    arr = [1, 2, 3, 4, 5]
    print(list(arr
               | where(lambda x: x % 2 == 0)
               | select(lambda x: x * 2)
    ))

if __name__ == "__main__":
    main()
