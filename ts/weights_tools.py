# -*- coding: utf-8 -*-

# ***************************************************
# * File        : weights_tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-15
# * Version     : 1.0.081510
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
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.switch_backend("agg")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
