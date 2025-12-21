# -*- coding: utf-8 -*-

# ***************************************************
# * File        : optimizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-13
# * Version     : 1.0.091321
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
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def select_optimizer(model, learning_rate: float, weight_decay: float, optimizer_type: str="adamw"):
    """
    optimizer
        - AdamW: 
        - GaLoreAdamW: https://github.com/jiaweizzhao/GaLore
    """
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = learning_rate, 
            weight_decay = weight_decay,
            fused=True,
        )
    elif optimizer_type == "galore_adamw":
        # define param groups as galore_params and non_galore_params
        # param_groups = [
        #     {
        #         'params': non_galore_params,
        #     }, 
        #     {
        #         'params': galore_params, 
        #         'rank': 128, 
        #         'update_proj_gap': 200, 
        #         'scale': 0.25, 
        #         'proj_type': 'std'
        #     }
        # ]
        optimizer = GaLoreAdamW(
            model.parameters(), 
            lr = learning_rate, 
            weight_decay=weight_decay,
        )

    return optimizer




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
