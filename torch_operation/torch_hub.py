# -*- coding: utf-8 -*-

# ***************************************************
# * File        : torch_hub.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
# * Description : torch.hub.list()
# *               torch.hub.help()
# *               torch.hub.load()
# * Link        : * https://pytorch.org/hub/
# *             : * https://pytorch.org/docs/stable/hub.html
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.hub as hub

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]





# 测试代码 main 函数
def main():
    torch.hub.list()
    torch.hub.help()
    torch.hub.load()

if __name__ == "__main__":
    main()
