#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************

# 条件判断
if [ 'test01' = 'test' ]; then
echo '等于'
fi


if [ 20 -gt 10 ]; then
echo '大于'
fi


if [ 10 -lt 20 ]; then
echo '小于'
fi


if [ -e /root/shell/a.txt ]; then
echo '存在'
fi


if [ 'test02' = 'test02' && echo 'hello' || echo 'world' ]; then
echo '条件满足，执行后面的语句'
fi


# 流程控制
if [ $1 -ge 60 ]
then
    echo 及格
elif [ $1 -lt 60 ]
then
    echo "不及格" 
fi


case $1 in
"1")
echo 周一
;;
"2")
echo 周二
;;
*)
echo 其它
;;
esac

