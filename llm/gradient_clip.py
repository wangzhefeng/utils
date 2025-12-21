# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gradient_clip.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-18
# * Version     : 1.0.091816
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def gradient_clipping(model, global_step, warmup_steps):
    """
    在预热阶段后应用梯度裁剪，防止梯度爆炸
    """
    if global_step >= warmup_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

    return model





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
