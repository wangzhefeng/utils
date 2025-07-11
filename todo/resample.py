# -*- coding: utf-8 -*-

# ***************************************************
# * File        : resample.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101716
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import VarianceThreshold

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def smote(data, tag_label = "tag_1", amount_personal = 0, std_rate = 5, k = 5, method = "mean"):
    """
    # smote unblance dataset
    """
    # cnt = data[tag_label].groupby(data[tag_label]).count()
    cnt = pd.value_counts(data[tag_label], sort = True).sort_index()
    rate = max(cnt) / min(cnt)
    location = []
    if rate < 5:
        print("不需要smote过程!")
        return data
    else:
        # 拆分不同大小的数据集合
        less_data = np.array(data[data[tag_label] == np.array(cnt[cnt == min(cnt)].index)[0]])
        more_data = np.array(data[data[tag_label] == np.array(cnt[cnt == max(cnt)].index)[0]])
        # 找出每个少量数据中每条数据k个临近点
        neighbors = NearestNeighbors(n_neighbors = k).fit(less_data)
        for i in range(len(less_data)):
            point = less_data[i, :]
            location_set = neighbors.kneighbors([less_data[i]], return_distance = False)[0]
            location.append(location_set)

        # 确定需要将少量数据补充到上限额度
        # 判断有没有设定生成数据个数, 如果没有按照std_rate(预期正负样本比)比例生成
        if amount_personal > 0:
            amount = amount_personal
        else:
            amount = int(max(cnt) / std_rate)

        # 初始化
        times = 0
        continue_index = []
        class_index = []
        for i in range(less_data.shape[1]):
            if len(pd.DataFrame(less_data[:, i]).drop_duplicates()) > 10:
                continue_index.append(i)
            else:
                class_index.append(i)
        case_update = pd.DataFrame()
        while times < amount:
            # 连续变量取附近k个点的重心, 认为少数样本的附近也是少数样本
            new_case = []
            pool = np.random.permutation(len(location))[0]
            neighbor_group = less_data[location[pool], :]
            if method == "mean":
                new_case1 = neighbor_group[:, continue_index].mean(axis = 0)
            # 连续样本的附近点向量上的点也是异常点
            if method == "random":
                new_case1 = less_data[pool][continue_index] + np.random.rand() * (less_data[pool][continue_index] - neighbor_group[0][continue_index])
            # 分类变量取mode
            new_case2 = []
            for i in class_index:
                L = pd.DataFrame(neighbors_group[:, i])
                new_case2.append(np.array(L.mode()[0])[0])
            new_case.extend(new_case1)
            new_case.extend(new_case2)
            case_update = pd.concat([case_update, pd.DataFrame(new_case)], axis = 1)
            print("已经生成了%s条数据, 完成百分之%.2f" % (times, times * 100 / amount))
            times += 1
        data_res = np.vstack((more_data, np.array(case_update.T)))
        data_res = pd.DataFrame(data_res)
        data_res.columns = data.columns
    return data_res


class sample_s(object):

    def __init__(self):
        pass

    def group_sample(self, data_set, label, percent=0.1):
        # 分层抽样
        # data_set:数据集
        # label:分层变量
        # percent:抽样占比
        # q:每次抽取是否随机,null为随机
        # 抽样根据目标列分层, 自动将样本数较多的样本分层按percent抽样, 得到目标列样本较多的特征欠抽样数据
        x = data_set
        y = label
        z = percent
        diff_case = pd.DataFrame(x[y]).drop_duplicates([y])
        result = []
        result = pd.DataFrame(result)
        for i in range(len(diff_case)):
            k = np.array(diff_case)[i]
            data_set = x[x[y] == k[0]]
            nrow_nb = data_set.iloc[:, 0].count()
            data_set.index = range(nrow_nb)
            index_id = rd.sample(range(nrow_nb), int(nrow_nb * z))
            result = pd.concat([result, data_set.iloc[index_id, :]], axis=0)
        new_data = pd.Series(result['label']).value_counts()
        new_data = pd.DataFrame(new_data)
        new_data.columns = ['cnt']
        k1 = pd.DataFrame(new_data.index)
        k2 = new_data['cnt']
        new_data = pd.concat([k1, k2], axis=1)
        new_data.columns = ['id', 'cnt']
        max_cnt = max(new_data['cnt'])
        k3 = new_data[new_data['cnt'] == max_cnt]['id']
        result = result[result[y] == k3[0]]
        return result

    def under_sample(self, data_set, label, percent=0.1, q=1):
        # 欠抽样
        # data_set:数据集
        # label:抽样标签
        # percent:抽样占比
        # q:每次抽取是否随机
        # 抽样根据目标列分层, 自动将样本数较多的样本按percent抽样, 得到目标列样本较多特征的欠抽样数据
        x = data_set
        y = label
        z = percent
        diff_case = pd.DataFrame(pd.Series(x[y]).value_counts())
        diff_case.columns = ['cnt']
        k1 = pd.DataFrame(diff_case.index)
        k2 = diff_case['cnt']
        diff_case = pd.concat([k1, k2], axis=1)
        diff_case.columns = ['id', 'cnt']
        max_cnt = max(diff_case['cnt'])
        k3 = diff_case[diff_case['cnt'] == max_cnt]['id']
        new_data = x[x[y] == k3[0]].sample(frac=z, random_state=q, axis=0)
        return new_data

    def combine_sample(self, data_set, label, number, percent=0.35, q=1):
        # 组合抽样
        # data_set:数据集
        # label:目标列
        # number:计划抽取多类及少类样本和
        # percent: 少类样本占比
        # q:每次抽取是否随机
        # 设定总的期待样本数量, 及少类样本占比, 采取多类样本欠抽样, 少类样本过抽样的组合形式
        x = data_set
        y = label
        n = number
        p = percent
        diff_case = pd.DataFrame(pd.Series(x[y]).value_counts())
        diff_case.columns = ['cnt']
        k1 = pd.DataFrame(diff_case.index)
        k2 = diff_case['cnt']
        diff_case = pd.concat([k1, k2], axis=1)
        diff_case.columns = ['id', 'cnt']
        max_cnt = max(diff_case['cnt'])
        k3 = diff_case[diff_case['cnt'] == max_cnt]['id']
        k4 = diff_case[diff_case['cnt'] != max_cnt]['id']
        n1 = p * n
        n2 = n - n1
        fre1 = n2 / float(x[x[y] == k3[0]]['label'].count())
        fre2 = n1 / float(x[x[y] == k4[1]]['label'].count())
        fre3 = ma.modf(fre2)
        new_data1 = x[x[y] == k3[0]].sample(frac=fre1, random_state=q, axis=0)
        new_data2 = x[x[y] == k4[1]].sample(frac=fre3[0], random_state=q, axis=0)
        test_data = pd.DataFrame([])
        if int(fre3[1]) > 0:
            i = 0
            while i < (int(fre3[1])):
                data = x[x[y] == k4[1]]
                test_data = pd.concat([test_data, data], axis=0)
                i += 1
        result = pd.concat([new_data1, new_data2, test_data], axis=0)
        return result


def var_filter(data, k = None):
    """
    方差选择法
    """
    var_data = data.var().sort_values()
    if k is not None:
        new_data = VarianceThreshold(threshold = k).fit_transform(data)
        return var_data, new_data
    else:
        return var_data


def pearson_value(data, label, k = None):
    """
    线性相关系数衡量
    """
    lable = str(label)
    # k为想删除的feature个数
    Y = data[label]
    x = data[[x for x in data.columns if x != label]]
    res = []
    for i in range(x.shape[1]):
        data_res = np.c_[Y, x.iloc[:, i]].T
        cor_value = np.abs(np.corrcoef(data_res)[0, 1])
        res.append([label, x.columns[i], cor_value])
    res = sorted(np.array(res), key = lambda x: x[2])
    if k is not None:
        if k < len(res):
            new_c = []
            for i in range(len(res) - k):
                new_c.append(res[i][1])
            return res, new_c
        else:
            print("feature个数越界!")
    else:
        return res


def vif_test(data, label, k = None):
    lable = str(label)
    # k为想删除的 feature 个数
    x = data[[x for x in data.colunms if x != label]]
    res = np.abs(np.corrcoef(x.T))
    vif_value = []
    for i in range(res.shape[0]):
        for j in range(res.shape[0]):
            if j > I:
                vif_value.append([x.columns[i], x.columns[j], res[i, j]])
    vif_value = sorted(vif_value, key = lambda x: x[2])
    if k is not None:
        if k < len(vif_value):
            new_c = []
            for i in range(len(x)):
                if vif_value[-i][1] not in new_c:
                    new_c.append(vif_value[-i][1])
                else:
                    new_c.append(vif_value[-i][0])
                if len(new_c) == k:
                    break
            out = [x for x in x.columns if x not in new_c]
            return vif_value, out
        else:
            print("feature个数越界!")
    else:
        return vif_value




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
