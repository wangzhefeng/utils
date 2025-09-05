#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************


# 系统函数
basename  # 删掉路径最后一个 '/'' 前的所有部分（包括 '/'），常用于获取文件名
basename [pathname] [suffix]
basename [string] [suffix]

basename /usr/bin/sort
basename include/stdio.h
basename include/stdio.h .h

dirname  # 删掉路径最后一个 '/' 后的所有部分（包括 '/'），常用于获取文件路径
dirname /usr/bin/
dirname dir1/str dir2/str
dirname stdio.h


# 自定义函数
function getSum(){
    SUM=$[$n1+$n2]
    echo "sum=$SUM"
}
read -p "请输入第一个参数 n1:" n1
read -p "请输入第二个参数 n2:" n2
getSum $n1 $n2  # 调用 getSum 函数
