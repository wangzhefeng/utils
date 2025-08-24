# -*- coding: utf-8 -*-

# ***************************************************
# * File        : option_set.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-25
# * Version     : 1.0.082500
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]




# 测试代码 main 函数
def main():
    torch.manual_seed(123)

    # create 2 training examples with 5 dimensions (features) each
    batch_example = torch.randn(2, 5) 
    
    # layers
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    logger.info(f"out: \n{out}")
    # out mean & var
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    logger.info(f"Mean: \n{mean}")
    logger.info(f"Variance: \n{var}")

    # layer norm
    out_norm = (out - mean) / torch.sqrt(var)
    logger.info(f"Normalized layer outputs: \n{out_norm}")
    # out mean & var
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    logger.info(f"Mean: \n{mean}")
    logger.info(f"Variance: \n{var}")
    torch.set_printoptions(sci_mode=False)
    logger.info(f"Mean: \n{mean}")
    logger.info(f"Variance: \n{var}")

if __name__ == "__main__":
    main()
