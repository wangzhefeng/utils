#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************

# 1.当前 shell 中所有变量: set
# set


# 2.删除变量
# unset $var


# 3.系统变量：
echo $HOME
echo $PWD
echo "User: $USER"


# 4.自定义变量
var1="hello world1"  # 定义变量
echo $var1  # 使用变量
unset $var1  # 删除变量
readonly var2="hello world2"  # 声明静态变量
echo $var2


# 5.将命令返回赋给变量
A=`ls`  # 反引号，执行里面的命令
A=$(ls)  # 等价于反引号


# 6.环境变量 
# export JAVA_HOME="usr/jdk1.8"  # 配置环境变量
# source /etc/profile  # 让环境变量生效
# echo $JAVA_HOME  # 引用环境变量


# 7.位置参数变量
echo $0  # 代表命令本身
echo $1  # 代表 1~9 个参数
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8
echo $9
echo ${10}  # 10以上参数用花括号
echo $*  # 命令行中所有参数，且把所有参数看成一个整体
echo $@  # 命令行中所有参数，且把每个参数区分对待
echo $#  # 所有参数个数
echo num_args=$#


# 8.预定义变量
echo $$  # 当前进程的 PID 进程号
echo $!  # 后台运行的最后一个进程的 PID 进程号
echo $?  # 最后一次执行的命令的返回状态，0为执行正确，非0执行失败

