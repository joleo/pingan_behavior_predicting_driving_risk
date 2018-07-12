# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_y
   Description :
   Author :       Administrator
   date：          2018/5/14 0014
-------------------------------------------------
   Change Activity:
                   2018/5/14 0014:
-------------------------------------------------
"""
__author__ = 'Administrator'
import numpy as np
import math
import pandas as pd

def R2(y_test, y_true):
    # 拟合优度
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

if __name__ == '__main__':
    data = pd.read_csv('../data/dm/train.csv')
    user_y = data.groupby(['TERMINALNO'], as_index=False)['Y'].first()
    pred = pd.read_csv('../model/20180526_lgb.csv')
    r2 = R2(pred['Pred'], user_y['Y'])
    print(r2)