# -*- coding: utf-8 -*-

# ***************************************************
# * File        : calc_loss.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

from utils.train_utils.train_funcs import select_criterion
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def calc_loss_batch(task_name, input_batch, target_batch, model, device):
    """
    calculate loss in batch training
    """
    # move data to device
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # criterion
    criterion = select_criterion()
    # logger.info(f"Train criterion has builded...")
    # Logits of last output token, loss
    if task_name == "tiny_gpt_classification_sft":
        logits = model(input_batch)[:, -1, :]
        loss = criterion(logits, target_batch)
    else:
        logits = model(input_batch)
        loss = criterion(logits.flatten(0, 1), target_batch.flatten())

    return loss


def calc_loss_loader(task_name, data_loader, model, device, num_batches=None):
    """
    calculate loss in all batches
    """
    # number of batches
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of 
        # batches in the data loader, if num_batches exceeds the number 
        # of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    # calculate loss
    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(task_name, input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches


def calc_loss(task, model, train_loader, valid_loader, test_loader, device):
    """
    计算训练、验证和测试集的损失

    Args:
        task (_type_): _description_
        model (_type_): _description_
        train_loader (_type_): _description_
        valid_loader (_type_): _description_
        test_loader (_type_): _description_
        device (_type_): _description_
    """
    # Disable gradient tracking for efficiency because we are not training, yet
    with torch.no_grad(): 
        train_loss = calc_loss_loader(task, train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(task, valid_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(task, test_loader, model, device, num_batches=5)
    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
