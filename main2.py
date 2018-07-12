# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main2
   Description :
   Author :       Administrator
   date：          2018/5/21 0021
-------------------------------------------------
   Change Activity:
                   2018/5/21 0021:
-------------------------------------------------
"""
__author__ = 'joleo'

import operator
import xgboost as xgb
from feature_integrate2 import *
# import warnings
from config import *

# warnings.filterwarnings('ignore')
fi = FeatureIntegrate()

start = time.time()

# 优化数据类型
train_dtypes = {'TERMINALNO': 'int32',
                'TIME': 'int32',
                'TRIP_ID': 'int16',
                'LONGITUDE': 'float32',
                'LATITUDE': 'float32',
                'DIRECTION': 'int16',
                'HEIGHT': 'float32',
                'SPEED': 'float32',
                'CALLSTATE': 'int8',
                'Y': 'float32'}

test_dtypes = {'TERMINALNO': 'int32',
                'TIME': 'int32',
                'TRIP_ID': 'int16',
                'LONGITUDE': 'float32',
                'LATITUDE': 'float32',
                'DIRECTION': 'int16',
                'HEIGHT': 'float32',
                'SPEED': 'float32',
                'CALLSTATE': 'int8'}

# 载入数据
train_data = pd.read_csv(path_train01, dtype=train_dtypes)
# 23,734,760
train_data = train_data.ix[:15000000, :]
test_data = pd.read_csv(path_test01, dtype=test_dtypes)

train = fi.train_feature_integrate(train_data)
test = fi.test_feature_integrate(test_data)


feature = [x for x in train.columns if x not in ['TERMINALNO','Y','hour_count_max','night_count_max'
                                                   ,'user_direction_std','call_unknow_state_per','user_call_num_per'
                                                    , 'user_lon_std', 'user_lon_mean'
                                                 ]]
print(feature)
print(train[feature].shape)
# print(train[feature].info())

param = {
    "objective": 'reg:linear',
    "eval_metric":'rmse',
    "seed":27,
    "booster": "gbtree",
    "min_child_weight":6,
    "gamma":0.1,
    # 'lambda':3,
    "max_depth": 5,
    "eta": 0.009,
    "silent": 1,
    "subsample":0.65,
    "colsample_bytree":0.35,
    "scale_pos_weight":0.9005
}

# ss_x = preprocessing.StandardScaler()
# train_x_disorder = ss_x.fit_transform(train[feature].fillna(-1))
# df_test = ss_x.transform(test[feature].fillna(-1))

train_y = train_data.groupby(['TERMINALNO'])['Y'].first()


df_train = xgb.DMatrix(train[feature].fillna(-1), train_y)
# df_train = xgb.DMatrix(train[feature].fillna(-1), train['Y'])
df_test = xgb.DMatrix(test[feature].fillna(-1))

model = xgb.train(param,df_train, num_boost_round=800)
y_pred = model.predict(df_test)


# feature importance
importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
print(list(df['feature']), list(df['fscore']))


# summission
id = test_data.groupby(['TERMINALNO'],as_index=False)['TRIP_ID'].first()
result = pd.DataFrame(id['TERMINALNO'])
result['Pred'] = y_pred
result = result.rename(columns={'TERMINALNO':'Id'})
result.loc[:, 'Pred'] = result['Pred']
result[['Id','Pred']].to_csv('model/result.csv',header=True,index=False)
print("cost time: " + str(time.time() - start))
