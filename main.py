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
from feature_integrate import *
from config import *
import warnings
warnings.filterwarnings('ignore')

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

param = {
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

df_train = xgb.DMatrix(train[feature].fillna(-1), train['Y'])
model = xgb.train(param,df_train, num_boost_round=800)

df_test = xgb.DMatrix(test[feature].fillna(-1))
y_pred = model.predict(df_test)


# feature importance
importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
print(list(df['feature']), list(df['fscore']))

# def modify(y):
#     if y>0 and y < 0.95: return y/2
#     elif y <1 : return y * 1.05
#     elif y < 1.5: return y *1.05
#     elif y < 100: return y/1.1
#     else: return y

# summission
result = pd.DataFrame(test['TERMINALNO'])
result['Pred'] = y_pred
result = result.rename(columns={'TERMINALNO':'Id'})
result.loc[:, 'Pred'] = result['Pred']#.map(lambda x: x+0.2 if x>0.6 else x)
result.to_csv('model/result.csv',header=True,index=False)
print("cost time: " + str(time.time() - start))
