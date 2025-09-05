# -*- coding: utf-8 -*-

# ***************************************************
# * File        : torch_test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-22
# * Version     : 1.0.082216
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

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger





# 测试代码 main 函数
def main():
    x = torch.arange(8).view(2, 2, 2)
    logger.info(f"x:\n{x}")
    
    logger.info(f"x.flip(dims=[0]):\n{x.flip(dims=[0])}")
    logger.info(f"x.flip(dims=[1]):\n{x.flip(dims=[1])}")
    logger.info(f"x.flip(dims=[2]):\n{x.flip(dims=[2])}")
    logger.info(f"x.flip(dims=[0, 1, 2]):\n{x.flip(dims=[0, 1, 2])}")

if __name__ == "__main__":
    main()
