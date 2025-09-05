#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************

# 运算符
# $((运算式))
# $[运算式]
# expr m + n  # 加法
# expr m - n  # 减法
# expr \*  # 乘法
# expr /  # 除法
# expr %  # 取余

# 第一种方式
echo $(((2+3)*4))

# 第二种方式
echo $[(2+3)*4]

# 使用 expr
TEMP=`expr 2 + 3`
echo `expr $TEMP \* 4`

