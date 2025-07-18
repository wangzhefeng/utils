# -*- coding: utf-8 -*-

# ***************************************************
# * File        : calc_dim_conv1d.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-07
# * Version     : 1.0.010723
# * Description : description
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def calc_dim_conv1d(in_channles: int, 
                    padding: int = 1,  
                    kernel_size: int = 3, 
                    stride: int = 1, 
                    dilation: int = 1):
    out_channels = (
        in_channles + \
        2 * padding - \
        dilation * (kernel_size - 1) \
            - 1
    ) / stride + 1
    
    return out_channels




# 测试代码 main 函数
def main():
    out_channels = calc_dim_conv1d(
        in_channles=1,
        padding=1,
        kernel_size=3,
        stride=1,
        dilation=1,
    )
    print(out_channels)

if __name__ == "__main__":
    main()
