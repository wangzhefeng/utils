# -*- coding: utf-8 -*-

# ***************************************************
# * File        : import_from_nb.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-04
# * Version     : 1.0.040414
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import io
import nbformat
import types
from typing import List

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def import_from_notebook(nb_fullname: str, func_names: List):
    """
    import module from jupyter notebook

    Args:
        nb_fullname (str): notebook full name
        func_names (List): function/class list in notebook
    """
    def import_definitions_from_notebook(fullname, names):
        current_dir = os.getcwd()
        path = os.path.join(current_dir, fullname + ".ipynb")
        path = os.path.normpath(path)

        # Load the notebook
        if not os.path.exists(path):
            raise FileNotFoundError(f"Notebook file not found at: {path}")

        with io.open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Create a module to store the imported functions and classes
        mod = types.ModuleType(fullname)
        sys.modules[fullname] = mod

        # Go through the notebook cells and only execute function or class definitions
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell_code = cell.source
                for name in names:
                    # Check for function or class definitions
                    if f"def {name}" in cell_code or f"class {name}" in cell_code:
                        exec(cell_code, mod.__dict__)
        return mod

    # fullname = "converting-gpt-to-llama2"
    # names = ["precompute_rope_params", "compute_rope", "SiLU", "FeedForward", "RMSNorm", "MultiHeadAttention"]

    return import_definitions_from_notebook(nb_fullname, func_names)




# 测试代码 main 函数
def main():
    # use case
    imported_module = import_from_notebook()

    demo_module = getattr(imported_module, "demo_module", None)

if __name__ == "__main__":
    main()
