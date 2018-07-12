# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       joleo
   date：          2018/5/8 0008
-------------------------------------------------
   Change Activity:
                   2018/5/8 0008:
-------------------------------------------------
"""
__author__ = 'joleo'
import operator
import xgboost as xgb
from model_test.model_xgb_012.feature_integrate import *
from model_test.model_xgb_012.config import *
fi = FeatureIntegrate()

start = time.time()

# 载入数据
train_data = pd.read_csv(path_train01)
train_data = train_data.ix[:15000000, :]
test_data = pd.read_csv(path_test01)


train_1 = fi.train_feature_integrate(train_data)
test_1 = fi.test_feature_integrate(test_data)

# feature_use = ['TRIP_ID', 'user_mean_speed', 'user_var_speed',
#        'user_max_speed', 'user_mean_height', 'user_var_height',
#        'user_mean_direction', 'hour_5_per', 'hour_6_per', 'hour_10_per',
#        'hour_15_per', 'hour_7_per', 'hour_8_per', 'hour_9_per', 'hour_17_per',
#        'hour_18_per', 'hour_11_per', 'hour_12_per', 'hour_13_per',
#        'hour_14_per', 'hour_0_per', 'hour_1_per', 'hour_2_per', 'hour_3_per',
#        'hour_4_per',  'hour_count_mean', 'hour_count_std',#'hour_count_max',
#        'hour_count_skew', 'night__day_delta', #'night_count_max',
#        'night_count_mean', 'night_count_std', 'night_count_skew',
#        'user_distance']

feature_use = [x for x in train_1.columns if x not in ['TERMINALNO','Y','hour_count_max','night_count_max'
                                                      ,'user_speed_skew','user_height_skew','direction_skew'
                                                      ]]
#
print(feature_use)
params = {
    "objective": 'reg:linear',
    "eval_metric":'rmse',
    "seed":27,
    "booster": "gbtree",
    "min_child_weight":6,
    "gamma":0.1,
    "max_depth": 5,
    "eta": 0.009,
    "silent": 1,
    "subsample":0.65,
    "colsample_bytree":0.35,
    "scale_pos_weight":0.9
}

train = xgb.DMatrix(train_1[feature_use].fillna(-1), train_1['Y'])
gbm = xgb.train(params,train,num_boost_round=800)

test = xgb.DMatrix(test_1[feature_use].fillna(-1))
pred = gbm.predict(test)

print(train_1[feature_use].shape)

# 特征重要度
importance = gbm.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
print(list(df['feature']), list(df['fscore']))

# 结果文件
result = pd.DataFrame(test_1['TERMINALNO'])
result['pre'] = pred
result = result.rename(columns={'TERMINALNO':'Id','pre':'Pred'})
result.loc[:, 'Pred'] = result['Pred']#.map(lambda x: abs(x) if x < 0 else x )
result.to_csv('../model/result_.csv',header=True,index=False)

print("cost time: " + str(time.time() - start))
