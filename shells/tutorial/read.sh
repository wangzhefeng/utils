#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************


read -p "请输入一个数 num1=" NUM1  # -p：指定读取值时的提示符
echo "你输入 num1 的值是: $NUM1"

read -t 10 -p "请在 10s 内输入一个数 num2=" NUM2
echo "你输入的 num2 的值是: $NUM2"

