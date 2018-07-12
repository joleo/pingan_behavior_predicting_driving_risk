# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     lgb_model
   Description :
   Author :       Administrator
   date：          2018/5/9 0009
-------------------------------------------------
   Change Activity:
                   2018/5/9 0009:
-------------------------------------------------
"""
__author__ = 'Administrator'

# -*- coding:utf8 -*-
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import lightgbm as lgb
from feature_extraction import *
import time

lyfe = LYFeatureExtraction()

path_train = "data/dm/train.csv"  # 训练文件
path_test = "data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def timestamp_datetime(value):
    fmt = '%Y-%m-%d %H:%M:%S'

    value = time.localtime(value)
    dt = time.strftime(fmt, value)
    return dt


def get_label(dataset, train):
    dataset['label'] = train.groupby(
        'TERMINALNO')['Y'].last().reset_index()['Y']
    return dataset


def save(test, pred, name):
    dt = datetime.datetime.now().strftime("%Y%m%d")
    test['Id'] = test['TERMINALNO']
    test['Pred'] = pred
    test[['Id', 'Pred']].to_csv(
        path_test_out + "%s_%s.csv" % (dt, name), index=False)


def fit_model(train, test, params, num_round, early_stopping_rounds):
    features = [x for x in train.columns if x not in ["TERMINALNO", 'label']]
    label = 'label'

    dtrain = lgb.Dataset(train[features], label=train[label])
    model = lgb.train(params, dtrain, num_boost_round=1500,

                      )
    t_pred = model.predict(test[features])
    return t_pred



if __name__ == "__main__":
    starttime = datetime.datetime.now()
    train_data = pd.read_csv(path_train)
    train_data = train_data.ix[:15000000, :]
    test_data = pd.read_csv(path_test)

    # 训练集
    user_driver_stat = lyfe.user_driver_stat(train_data)  # 用户驾驶行为基本特征统计
    user_driver_time = lyfe.user_driver_time(train_data)  # 用户时间特征特征
    user_night_stat = lyfe.user_night_stat(train_data)  # 夜晚开车和每天用车时长的统计
    get_distance = lyfe.get_distance(train_data)  # 经纬度转距离

    feature = pd.merge(user_driver_stat, user_driver_time, how='left', on='TERMINALNO')
    feature = pd.merge(feature, user_night_stat, how='left', on='TERMINALNO')
    feature = pd.merge(feature, get_distance, how='left', on='TERMINALNO')

    train = get_label(feature, train_data)

    # print("========building the dataset========")
    # 测试集
    user_driver_stat = lyfe.user_driver_stat(test_data)  # 用户驾驶行为基本特征统计
    user_driver_time = lyfe.user_driver_time(test_data)  # 用户时间特征特征
    user_night_stat = lyfe.user_night_stat(test_data)  # 夜晚开车和每天用车时长的统计
    get_distance = lyfe.get_distance(test_data)  # 经纬度转距离


    test = pd.merge(user_driver_stat, user_driver_time, how='left', on='TERMINALNO')
    test = pd.merge(test, user_night_stat, how='left', on='TERMINALNO')
    test = pd.merge(test, get_distance, how='left', on='TERMINALNO')


    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        # 'is_unbalance': 'True',
        'learning_rate': 0.01,
        # 'verbose': 1,
        'num_leaves': 2**5,
        'max_bin': 55,
        'objective': 'regression',
        'feature_fraction': 0.7,
        'bagging_fraction': 0.2319,
        # 'bagging_freq': 5,
        # 'subsample':0.45,
        'feature_fraction_seed' :9,
        'bagging_seed' : 9,
        # 'seed': 27,
        'nthread': 4,
        'min_data_in_leaf':6,
        'min_sum_hessian_in_leaf' : 11,
        'silent': True,
    }

    num_round = 700
    early_stopping_rounds = 100
    # print("============training model===========")
    rlt_pred = fit_model(train, test, params, num_round, early_stopping_rounds)

    save(test, rlt_pred, "lgb")
    endtime = datetime.datetime.now()
    print("use time: ", (endtime - starttime).seconds, " s")
