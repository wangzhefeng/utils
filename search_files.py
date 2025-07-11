# -*- coding: utf-8 -*-

# ***************************************************
# * File        : file_glob_search.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-23
# * Version     : 0.1.072301
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
import argparse
import glob
import fnmatch
import pathlib
from typing import List

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def search_folder(path : str, extension: str, file_size: List = None):
    """
    Search folder for files
    
    Args:
        path (str): file path
        extension (str): 一次搜索多个扩展名
        file_size (List, optional): 对文件大小的范围进行过滤. Defaults to None.
    """
    folder = pathlib.Path(path)
    files = list(folder.rglob(f"*.{extension}"))

    if not files:
        print(f"No files found with {extension}")
        return

    if file_size is not None:
        files = [
            file
            for file in files
            if file.stat().st_size >= file_size[0] and file.stat().st_size <= file_size[1]
        ]
    print(f"{len(files)} *.{extension} files found.")
    for file in files:
        print(file)


def arg_parser():
    parser = argparse.ArgumentParser(
        "PySearch",
        description = "PySearch - The Python Powered File Searcher",
        epilog = "Thank you for choosing PySearch!",
        # add_help = False,
    )
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument("-p", "--path", help = "The path to search for files", required = True, dest = "path")
    parser.add_argument("-e", "--exte", help = "The extension to search for", required = True, dest  = "extension")
    parser.add_argument("-s", "--size", help = "The file size to filter on in bytes", required = False, type = int, dest = "size", default = None)
    args = parser.parse_args()
    search_folder(args.path, args.extension, args.size)




# 测试代码 main 函数
def main():
    arg_parser()

if __name__ == "__main__":
    main()
