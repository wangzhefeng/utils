# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-12-21
# * Version     : 1.0.122120
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
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


import shutil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

# 显示当前 Matplotlib 使用的字体目录
print("字体缓存目录:", matplotlib.get_cachedir())
print("字体配置文件路径:", matplotlib.matplotlib_fname())
# 列出所有可用字体
for font in fontManager.ttflist:
    if 'simhei' in font.name.lower() or 'ming' in font.name.lower() or 'kai' in font.name.lower():
        print(f"字体名称: {font.name}, 路径: {font.fname}")



# 获取Matplotlib字体目录
font_dir = matplotlib.get_data_path() + "\\fonts\\ttf"
print("字体目录:", font_dir)

# 复制字体文件到Matplotlib字体目录（假设字体文件在C:\Windows\Fonts\simhei.ttf）
# shutil.copy("C:\\Windows\\Fonts\\Arial Unicode MS.ttf", font_dir)

# 清除字体缓存
# _rebuild()



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
