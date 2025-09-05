# -*- coding: utf-8 -*-

# ***************************************************
# * File        : torch_cuda.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-28
# * Version     : 1.0.082810
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger





# 测试代码 main 函数
def main():
    # Mix-Precision Training

    # 检查 GPU 是否支持 bfloat16(brain float point)
    import torch
    logger.info(torch.cuda.is_bf16_supported())

if __name__ == "__main__":
    main()
