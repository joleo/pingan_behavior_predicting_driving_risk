# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     rf_model
   Description :
   Author :       Administrator
   date：          2018/5/25 0025
-------------------------------------------------
   Change Activity:
                   2018/5/25 0025:
-------------------------------------------------
"""
__author__ = 'Administrator'
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import numpy as np
from feature_integrate import *
from config import *

fi = FeatureIntegrate()

start = time.time()

# 载入数据
train_data = pd.read_csv(path_train01)
# 23,734,760
train_data = train_data.ix[:15000000, :]
test_data = pd.read_csv(path_test01)

# train_data = train_data.loc[train_data['SPEED'] >= 0]
# test_data = test_data.loc[test_data['SPEED'] >= 0]

train = fi.train_feature_integrate(train_data)
test = fi.test_feature_integrate(test_data)

# , 'night__day_delta', 'night_count_skew','hour_count_skew'
#  'user_lat_std','user_lon_std','user_lat_mean', 'user_lon_mean'
feature = [x for x in train.columns if x not in ['TERMINALNO','Y','hour_count_max','night_count_max'
                                                   ,'user_direction_std','call_unknow_state_per','user_call_num_per'
                                                 ,'user_lat_std','user_lat_mean','user_lon_std', 'user_lon_mean'
                                                ,'night__day_delta', 'night_count_skew','hour_count_skew']]
print(feature)
print(train[feature].shape)
# param = {
#     "objective": 'reg:linear',
#     "eval_metric":'rmse',
#     "seed":27,
#     "booster": "gbtree",
#     "min_child_weight":6,
#     "gamma":0.1,
#     "max_depth": 5,
#     "eta": 0.009,
#     "silent": 1,
#     "subsample":0.65,
#     "colsample_bytree":0.35,
#     "scale_pos_weight":0.9
# }
#
# df_train = xgb.DMatrix(train[feature].fillna(-1), train['Y'])
# model = xgb.train(param,df_train, num_boost_round=800)
#
# df_test = xgb.DMatrix(test[feature].fillna(-1))
# y_pred = model.predict(df_test)
train_x = train[feature].fillna(-1)
train_y = train['Y']
test_x = test[feature].fillna(-1)
model = RandomForestRegressor(n_estimators=1000,criterion='mse',max_depth=5,max_features=0.55,min_samples_leaf=6,n_jobs=4,random_state=27)#
scores = cross_val_score(model,train_x.values,train_y.values,cv=5,scoring='mean_squared_error')
print(np.sqrt(-scores),np.mean(np.sqrt(-scores)))

model.fit(train_x.values,train_y.values)

y_pred = model.predict(test_x.values)

result = pd.DataFrame(test['TERMINALNO'])
result['Pred'] = y_pred
result = result.rename(columns={'TERMINALNO':'Id'})
result.loc[:, 'Pred'] = result['Pred']#.map(lambda x: abs(x))
result.to_csv('model/result.csv',header=True,index=False)
print("cost time: " + str(time.time() - start))