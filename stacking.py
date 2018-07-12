# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     stacking
   Description :
   Author :       Administrator
   date：          2018/5/10 0010
-------------------------------------------------
   Change Activity:
                   2018/5/10 0010:
-------------------------------------------------
"""
__author__ = 'Administrator'


from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from math import sqrt

import numpy as np
import scipy.stats as st
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
# from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import pandas as pd
import time
from vecstack import stacking
from config import *
from feature_integrate import *
from config import *

fi = FeatureIntegrate()

start = time.time()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def transformer(y, func=None):
    if func is None:
        return y
    else:
        return func(y)


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def stacking(models, X_train, y_train, X_test, regression=True,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=4, stratified=False,
             shuffle=False, random_state=0, verbose=0):
    # Print type of task
    if regression and verbose > 0:
        print('task:   [regression]')
    elif not regression and verbose > 0:
        print('task:   [classification]')

    # Specify default metric for cross-validation
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        metric = accuracy_score

    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)

    # Split indices to get folds (stratified can be used only for classification)
    if stratified and not regression:
        kf = StratifiedKFold(y_train, n_folds, shuffle=shuffle, random_state=random_state)

    else:
        kf = KFold(len(y_train), n_folds,shuffle=shuffle, random_state=random_state)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))

    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))

        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], len(kf)))

        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf):
            X_tr = X_train.loc[tr_index]
            y_tr = y_train.loc[tr_index]
            X_te = X_train.loc[te_index]
            y_te = y_train.loc[te_index]

            # Fit 1-st level model
            model = model.fit(X_tr, transformer(y_tr, func=transform_target))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(model.predict(X_te), func=transform_pred)
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(model.predict(X_test), func=transform_pred)

            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))

        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = np.mean(S_test_temp, axis=1)
        else:
            S_test[:, model_counter] = st.mode(S_test_temp, axis=1)[0].ravel()

        if verbose > 0:
            print('    ----')
            print('    MEAN:   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    return (S_train, S_test)


# X_train = tr_user[features].replace([np.inf, np.nan], 0).reset_index(drop=True)
# X_test = ts_user[features].replace([np.inf, np.nan], 0).reset_index(drop=True)
# y_train = tr_user["loan_sum"].reset_index(drop=True)
# 载入数据
train_data = pd.read_csv(path_train01)
train_data = train_data.ix[:15000000, :]
test_data = pd.read_csv(path_test01)


train = fi.train_feature_integrate(train_data)
test = fi.test_feature_integrate(test_data)

feature = [x for x in train.columns if x not in ['TERMINALNO','Y','hour_count_max','night_count_max']]
                                                 # ,'user_speed_skew','user_height_skew','direction_skew']]

X_train = train[feature].fillna(-1)
X_test = test[feature].fillna(-1)
y_train = train['Y']
print(X_train.shape) # (100, 36)

# Caution! All models and parameter values are just
# demonstrational and shouldn't be considered as recommended.
# Initialize 1-st level models.
models = [
    # BaggingRegressor(random_state=27, n_jobs=1, n_estimators=300),
    RandomForestRegressor(random_state=35, n_jobs=1,n_estimators=720, max_depth=5),
    # XGBRegressor(seed=0, learning_rate=0.01,n_estimators=720, max_depth=5),
    # LGBMRegressor(num_leaves=5, learning_rate=0.01, n_estimators=720),
    # GradientBoostingRegressor(learning_rate=0.01, subsample=0.5, max_depth=6, n_estimators=720)
    ]

# Compute stacking features

S_train, S_test = stacking(models, X_train, y_train, X_test, regression=True, metric=mean_squared_error, n_folds=5,
                           shuffle=True, random_state=5, verbose=2)

# Fit 2-nd level model
# model = LGBMRegressor(num_leaves=5, learning_rate=0.01, n_estimators=720)
model = LGBMRegressor(num_leaves=5,
                      learning_rate=0.01, n_estimators=720,
                      max_bin = 55, bagging_fraction = 0.8,
                      bagging_freq = 5, feature_fraction = 0.2319,
                      feature_fraction_seed=9, bagging_seed=9,
                      min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)

# 结果文件
result = pd.DataFrame(test['TERMINALNO'])
result['Pred'] = y_pred
result = result.rename(columns={'TERMINALNO':'Id'})
result.loc[:, 'Pred'] = result['Pred'].map(lambda x: abs(x))
result.to_csv('model/result.csv',header=True,index=False)

print("cost time: " + str(time.time() - start))
