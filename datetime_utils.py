# -*- coding: utf-8 -*-


# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-06
# * Version     : 0.1.070623
# * Description : description
# * Link        : https://docs.python.org/3.7/library/calendar.html
# *               https://docs.python.org/3.7/library/datetime.html#module-datetime
# * Requirement : 日期、时间相关的工具函数
# ***************************************************

# python libraries
import calendar
import datetime


def getMonthDays(year: int, month: int) -> int:
    """
    计算某年某月的天数
    """
    monthRange = calendar.monthrange(year, month)
    return monthRange[1]


def isLeapYear(year: int) -> bool:
    """
    判断某年是否是闰年
    """
    # method 1
    # if (year % 4) == 0 and (year % 100) != 0 or (year % 400) == 0:
    #     return True
    # else:
    #     return False
    # method 2
    if calendar.isleap(year):
        return True
    else:
        return False


def createCalendar(year: int, month: int):
    """
    生成某年某月的日历
    """
    print(calendar.month(year, month))


def getYesterday():
    """
    获取昨天的日期
    """
    today = datetime.date.today()
    one_day = datetime.timedelta(days = 1)
    yesterday = today - one_day
    return yesterday


def dayOfYear(year: int, month: int, day: int) -> int:
    """
    判断某一天是改年的第几天
    """
    one_day_of_year = datetime.date(year = int(year), month = int(month), day = int(day))
    start_of_year = datetime.date(year = int(year), month = 1, day = 1)
    return (one_day_of_year - start_of_year).days + 1


# 控制模块被全部导入的内容
__all__ = [
    "getMonthDays", 
    "isLeapYear", 
    "createCalendar", 
    "getYesterday", 
    "dayOfYear",
]




# 测试代码 main 函数
def main():
    # 计算某年某月的天数
    monthRange = getMonthDays(2022, 7)
    print(monthRange)

    # 判断某年是否是闰年
    is_leap_year = isLeapYear(2022)
    print(is_leap_year)

    # 生成某年某月的日历
    createCalendar(2022, 7)

    # 获取昨天的日期
    yesterday = getYesterday()
    print(yesterday)
    print(type(yesterday))

    # 获取某一天为该年的第几天
    day_of_year = dayOfYear(2022, 7, 6)
    print(day_of_year)

if __name__ == "__main__":
    main()
