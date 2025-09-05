# -*- coding: utf-8 -*-

# ***************************************************
# * File        : losses.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101716
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math

import numpy as np

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def loglikelihood(pred, train_data):
    """
    self-defined objective function
    f(preds: array, train_data: Dataset) -> grad: array, hess: array
    log likelihood loss

    Args:
        pred ([type]): [description]
        train_data ([type]): [description]
    """
    labels = train_data.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    
    return grad, hess


def logloss(y_true, y_pred):
    label2num = dict(
        (name, i) for i, name in enumerate(sorted(set(y_true)))
    )
    return -1 * sum(
        math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf 
        for y, label in zip(y_pred, y_true)
    ) / len(y_pred)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
