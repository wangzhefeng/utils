# -*- coding: utf-8 -*-

# ***************************************************
# * File        : BayesOptim.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101718
# * Description : description
# * Link        : https://github.com/bayesian-optimization/BayesianOptimization
# *               https://bayesian-optimization.github.io/BayesianOptimization/1.5.1/
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from bayes_opt import BayesianOptimization

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# the function to be optimized
# ------------------------------
def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


# ------------------------------
# start
# ------------------------------
# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)

print(optimizer.max)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))


# ------------------------------
# changing bounds
# ------------------------------
optimizer.set_bounds(new_bounds={"x": (-2, 3)})

optimizer.maximize(
    init_points=0,
    n_iter=5,
)

# ------------------------------
# 
# ------------------------------


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
