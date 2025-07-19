#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************

# for 循环
# --------------------
# 使用$* 
for i in "$*" 
do     
    echo "the arg is $i" 
done 


# 使用$@ 
for j in "$@" 
do     
    echo "the arg is $j" 
done


# 输出从 1 加到 100 的值
SUM=0
for ((i=1;i<=100;i++))
do
    SUM=$[SUM+$i]
done
echo $SUM

# while 循环
# --------------------
SUM=0
i=0
while [ $i -le $1 ]
do
    SUM=$[$SUM+$i]
    i=$[$i+1]
done
echo $SUM

