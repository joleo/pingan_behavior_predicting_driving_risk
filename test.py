# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       Administrator
   date：          2018/5/8 0008
-------------------------------------------------
   Change Activity:
                   2018/5/8 0008:
-------------------------------------------------
"""
__author__ = 'Administrator'
# from __future__ import division
import numpy as np
import pandas as pd
import random
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from random import randint
import operator
from feature_integrate import *
# import warnings
from config import *

# warnings.filterwarnings('ignore')
fi = FeatureIntegrate()



# from xgboost.sklearn import XGBClassifiers


'''
群体大小，一般取20~100；终止进化代数，一般取100~500；交叉概率，一般取0.4~0.99；变异概率，一般取0.0001~0.1。
'''
# generations = 400   # 繁殖代数 100
pop_size = 20  # 种群数量  500
# max_value = 10      # 基因中允许出现的最大值 （可防止离散变量数目达不到2的幂的情况出现，限制最大值，此处不用）
chrom_length = 15  # 染色体长度
pc = 0.6  # 交配概率
pm = 0.01  # 变异概率
results = []  # 存储每一代的最优解，N个三元组（auc最高值, n_estimators, max_depth）
fit_value = []  # 个体适应度
fit_mean = []  # 平均适应度
# pop = [[0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] for i in range(pop_size)] # 初始化种群中所有个体的基因初始序列

random_seed = 20
cons_value = 0.19 / 31  # (0.20-0.01）/ (32 - 1)

'''要调试的参数有：（参考：http://xgboost.readthedocs.io/en/latest/parameter.html）
   tree_num：基树的棵数   ----------------（要调的参数）
   eta: 学习率（learning_rate），默认值为0.3，范围[0,1]  ----------------（要调的参数）
   max_depth: 最大树深，默认值为6   ----------------（要调的参数）
   min_child_weight：默认值为1，范围[0, 正无穷]，该参数值越小，越容易 overfitting，当它的值较大时，可以避免模型学习到局部的特殊样本。 ----------（要调的参数）
   gamma：默认值为0，min_split_loss，范围[0, 正无穷]
   subsample：选择数据集百分之多少来训练，可以防止过拟合。默认值1，范围(0, 1]，理想值0.8
   colsample_bytree：subsample ratio of columns when constructing each tree，默认值1，范围(0, 1]，理想值0.8，太小的值会造成欠拟合
   lambda：L2 regularization term on weights, increase this value will make model more conservative.参数值越大，模型越不容易过拟合
   alpha：L1 regularization term on weights, increase this value will make model more conservative.参数值越大，模型越不容易过拟合

   上述参数，要调的有4个，其他的采用理想值就可以
   tree_num: [10、 20、 30、......150、160] 用4位二进制, 0000代表10
   eta: [0.01, 0.02, 0.03, 0.04, 0.05, ...... 0.19, 0.20]   0.2/0.01=20份，用5位二进制表示足够（2的4次方<20<2的5次方）
       00000 -----> 0.01
       11111 -----> 0.20
       0.01 + 对应十进制*（0.20-0.01）/ (2的5次方-1)
   max_depth:[3、4、5、6、7、8、9、10]   用3位二进制
   min_child_weight: [1, 2, 3, 4, 5, 6, 7, 8]  用3位二进制

   示例：   0010,         01001,               010,      110  （共15位）
         tree_num         eta               max_depth  min_child_weight
        (1+2)*10=30  0.01+9*0.005939=0.06       3+2=5      1+6=7
'''


def xgboostModel(tree_num, eta, max_depth, min_child_weight, random_seed):

    # train_xy = loadFile("../../Data/train-gao.csv")
    # train_xy = train_xy.drop('ID', axis=1)  # 删除训练集的ID
    # # 将训练集划分成8:2（训练集与验证集比例）的比例
    # train, val = train_test_split(
    #     train_xy, test_size=0.2, random_state=80)
    #
    # train_y = train.Kind
    # train_x = train.drop('Kind', axis=1)
    # dtrain = xgb.DMatrix(train_x, label=train_y)
    #
    # val_y = val.Kind
    # val_x = val.drop('Kind', axis=1)
    # dval = xgb.DMatrix(val_x)
    # 载入数据
    train_data = pd.read_csv(path_train01)
    # 23,734,760
    train_data = train_data.ix[:15000000, :]
    # test_data = pd.read_csv(path_test01)
    train = fi.train_feature_integrate(train_data)
    # test = fi.test_feature_integrate(test_data)
    train, val = train_test_split(train, test_size=0.2, random_state=80)

    feature = [x for x in train.columns if x not in ['TERMINALNO', 'Y', 'hour_count_max', 'night_count_max'
        , 'user_direction_std', 'call_unknow_state_per', 'user_call_num_per'
        , 'user_lon_std', 'user_lon_mean'
                                                     ]]
    df_train = xgb.DMatrix(train[feature].fillna(-1), train['Y'])
    # df_test = xgb.DMatrix(test[feature].fillna(-1))
    val_x = val.drop('Y', axis=1)
    df_val = xgb.DMatrix(val_x[feature].fillna(-1))
    val_y = val.Y
    params = {
        'booster': 'gbtree',  # gbtree used
        'objective':  'reg:linear',
        'early_stopping_rounds': 100,
        # 'scale_pos_weight': 0.13,  # 正样本权重
        'eval_metric': 'rmse',
        'eta': eta,  # 0.02
        'max_depth': max_depth,  # 8
        'min_child_weight': min_child_weight,  # 3
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 50,
        'alpha': 19,
        'seed': randint(1, 100),
        'nthread': 4,
        'silent': 1
    }
    model = xgb.train(params, df_train, num_boost_round=tree_num)
    predict_y = model.predict(df_val, ntree_limit=model.best_ntree_limit)
    # roc_auc = metrics.roc_auc_score(val_y, predict_y)
    rmse = metrics.mean_squared_error(val_y, predict_y)
    # metrics
    return rmse



def loadFile(filePath):
    fileData = pd.read_csv(filePath)
    return fileData


# Step 1 : 对参数进行编码（用于初始化基因序列，可以选择初始化基因序列，本函数省略）
def geneEncoding(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]


# Step 2 : 计算个体的目标函数值
def cal_obj_value(pop):
    objvalue = []
    variable = decodechrom(pop)
    for i in range(len(variable)):
        tempVar = variable[i]

        tree_num_value = (tempVar[0] + 1) * 10
        eta_value = 0.01 + tempVar[1] * cons_value
        max_depth_value = 3 + tempVar[2]
        min_child_weight_value = 1 + tempVar[3]

        aucValue = xgboostModel(tree_num_value, eta_value, max_depth_value, min_child_weight_value, random_seed)
        objvalue.append(aucValue)
    return objvalue  # 目标函数值objvalue[m] 与个体基因 pop[m] 对应


# 对每个个体进行解码，并拆分成单个变量，返回 tree_num（4）、eta（5）、max_depth（3）、min_child_weight（3）
def decodechrom(pop):
    variable = []
    for i in range(len(pop)):
        res = []

        # 计算第一个变量值，即 0101->10(逆转)
        temp1 = pop[i][0:4]
        v1 = 0
        for i1 in range(4):
            v1 += temp1[i1] * (math.pow(2, i1))
        res.append(int(v1))

        # 计算第二个变量值
        temp2 = pop[i][4:9]
        v2 = 0
        for i2 in range(5):
            v2 += temp2[i2] * (math.pow(2, i2))
        res.append(int(v2))

        # 计算第三个变量值
        temp3 = pop[i][9:12]
        v3 = 0
        for i3 in range(3):
            v3 += temp3[i3] * (math.pow(2, i3))
        res.append(int(v3))

        # 计算第四个变量值
        temp4 = pop[i][12:15]
        v4 = 0
        for i4 in range(3):
            v4 += temp4[i4] * (math.pow(2, i4))
        res.append(int(v4))

        variable.append(res)
    return variable


# Step 3: 计算个体的适应值（计算最大值，于是就淘汰负值就好了）
def calfitvalue(obj_value):
    fit_value = []
    temp = 0.0
    Cmin = 0
    for i in range(len(obj_value)):
        if (obj_value[i] + Cmin > 0):
            temp = Cmin + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value


# Step 4: 找出适应函数值中最大值，和对应的个体
def best(pop, fit_value):
    best_individual = pop[0]
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]


# Step 5: 每次繁殖，将最好的结果记录下来(将二进制转化为十进制)
def b2d(best_individual):
    # 计算第一个变量值
    temp1 = best_individual[0:4]
    v1 = 0
    for i1 in range(4):
        v1 += temp1[i1] * (math.pow(2, i1))
    v1 = (v1 + 1) * 10

    # 计算第二个变量值
    temp2 = best_individual[4:9]
    v2 = 0
    for i2 in range(5):
        v2 += temp2[i2] * (math.pow(2, i2))
    v2 = 0.01 + v2 * cons_value

    # 计算第三个变量值
    temp3 = best_individual[9:12]
    v3 = 0
    for i3 in range(3):
        v3 += temp3[i3] * (math.pow(2, i3))
    v3 = 3 + v3

    # 计算第四个变量值
    temp4 = best_individual[12:15]
    v4 = 0
    for i4 in range(3):
        v4 += temp4[i4] * (math.pow(2, i4))
    v4 = 1 + v4

    return int(v1), float(v2), int(v3), int(v4)


# Step 6: 自然选择（轮盘赌算法）
def selection(pop, fit_value):
    # 计算每个适应值的概率
    new_fit_value = []
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        new_fit_value.append(fit_value[i] / total_fit)
    # 计算每个适应值的累积概率
    cumsum(new_fit_value)
    # 生成随机浮点数序列
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    # 对生成的随机浮点数序列进行排序
    ms.sort()
    # 轮盘赌算法（选中的个体成为下一轮，没有被选中的直接淘汰，被选中的个体代替）
    fitin = 0
    newin = 0
    newpop = pop
    while newin < pop_len:
        if (ms[newin] < new_fit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop


# 求适应值的总和
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total


# 计算累积概率
def cumsum(fit_value):
    temp = []
    for i in range(len(fit_value)):
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j = j + 1
        temp.append(t)
    for i in range(len(fit_value)):
        fit_value[i] = temp[i]


# Step 7: 交叉繁殖
def crossover(pop, pc):  # 个体间交叉，实现基因交换
    poplen = len(pop)
    for i in range(poplen - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0: cpoint])
            temp1.extend(pop[i + 1][cpoint: len(pop[i])])
            temp2.extend(pop[i + 1][0: cpoint])
            temp2.extend(pop[i][cpoint: len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2


# Step 8: 基因突变
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


def writeToFile(var, w_path):
    # f = file(w_path, "a+")
    with open(w_path, 'r') as f:
        for item in var:
            f.write(str(item) + "\r\n")
        f.close()


def generAlgo(generations):
    pop = geneEncoding(pop_size, chrom_length)
    print(str(generations) + " start...")
    for i in range(generations):
        # print("第 " + str(i) + " 代开始繁殖......")
        obj_value = cal_obj_value(pop)  # 计算目标函数值
        # print(obj_value)
        fit_value = calfitvalue(obj_value)  # 计算个体的适应值
        # print(fit_value)
        [best_individual, best_fit] = best(pop, fit_value)  # 选出最好的个体和最好的函数值
        # print("best_individual: "+ str(best_individual))
        v1, v2, v3, v4 = b2d(best_individual)
        results.append([best_fit, v1, v2, v3, v4])  # 每次繁殖，将最好的结果记录下来
        # print(str(best_individual) + " " + str(best_fit))
        selection(pop, fit_value)  # 自然选择，淘汰掉一部分适应性低的个体
        crossover(pop, pc)  # 交叉繁殖
        mutation(pop, pc)  # 基因突变
    # print(results)
    results.sort()
    # wirte results to file
    writeToFile(results, "generation_" + str(generations) + ".txt")
    print(results[-1])
    # print(xgboostModel(100, 12))


if __name__ == '__main__':
    # gen = [100, 200, 300, 400, 500]
    gen = [10, 20, 30, 40, 50]
    for g in gen:
        generAlgo(int(g))
    # pop = geneEncoding(pop_size, chrom_length)
    # for i in range(generations):
    #     print("第 " + str(i) + " 代开始繁殖......")
    #     obj_value = cal_obj_value(pop) # 计算目标函数值
    #     # print(obj_value)
    #     fit_value = calfitvalue(obj_value); #计算个体的适应值
    #     # print(fit_value)
    #     [best_individual, best_fit] = best(pop, fit_value) #选出最好的个体和最好的函数值
    #     # print("best_individual: "+ str(best_individual))
    #     v1, v2, v3, v4 = b2d(best_individual)
    #     results.append([best_fit, v1, v2, v3, v4]) #每次繁殖，将最好的结果记录下来
    #     print(str(best_individual) + " " + str(best_fit))
    #     selection(pop, fit_value) #自然选择，淘汰掉一部分适应性低的个体
    #     crossover(pop, pc) #交叉繁殖
    #     mutation(pop, pc) #基因突变
    # # print(results)
    # results.sort()
    # # wirte results to file
    # writeToFile(results, "generation_" + str(generations) + ".txt")
    # print(results[-1])
    # # print(xgboostModel(100, 12))

