# -*- coding: utf-8 -*-

# ***************************************************
# * File        : plotly_utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-07-11
# * Version     : 0.1.071123
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

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def scatter_3d_plot(df, x, y, z):
    fig = go.Figure(
        data = [go.Scatter3d(
            x = df[x],
            y = df[y],
            z = df[z],
            mode = 'markers',
            marker = dict(
                size = 3,
                # color = z,  # set color to an array/list of desired values
                # colorscale = 'Viridis',  # choose a colorscale
                opacity = 0.8
            )
        )]
    )
    # tight layout
    fig.update_layout(margin = dict(l = 0, r = 0, b = 0, t = 0))
    fig.show()


def scatter_3d_plot(df, x, y, z):
    fig = plt.figure(figsize = (20, 8))
    # Plotly Express
    fig = px.scatter_3d(
        df, 
        x = x, 
        y = y, 
        z = z,
    )
    # tight layout
    fig.update_layout(margin = dict(l = 3, r = 3, b = 3, t = 3))
    fig.show()


def scatter_3d_plot(x: np.array, y: np.ndarray, z: np.ndarray):
    fig = go.Figure(data=[go.Scatter3d(
        x = x,
        y = y, 
        z = z,  
        mode = 'markers',
        marker = dict(
            size = 3,
            # color=z,                # set color to an array/list of desired values
            # colorscale='Viridis',   # choose a colorscale
            opacity = 0.8
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
