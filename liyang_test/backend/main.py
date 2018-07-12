# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     xgb_model
   Description :
   Author :       liyang
   date：          2018/5/7 0007
-------------------------------------------------
   Change Activity:
                   2018/5/7 0007:
-------------------------------------------------
"""
__author__ = 'liyang'

import xgboost as xgb
import warnings

from ly_feature_integrate import *
lyfi = LYFeatureIntegrate()
warnings.filterwarnings('ignore')

start = time.time()

all_features_list,user_Y_list,min_Y,max_Y = lyfi.get_all_features()
userid_list,test_features = lyfi.get_test_features02()

train = np.array(all_features_list)
label = np.array(user_Y_list).T
test = np.array(test_features)
id = np.array(userid_list).T

# model_xgb_1_2018_5_8
# params = {
#     "objective": 'reg:linear',
#     "eval_metric":'rmse',
#     "seed":110,
#     "booster": "gbtree",
#     "min_child_weight":5,
#     "gamma":0.1,
#     "max_depth": 4,
#     "eta": 0.01,
#     "silent": 1,
#     "subsample":0.75,
#     "colsample_bytree":0.45,
#     "scale_pos_weight":0.9
# }

# model_xgb_3_2018_5_8
# params = {
#     "objective": 'reg:linear',
#     "eval_metric":'rmse',
#     "seed":70,
#     "booster": "gbtree",
#     "min_child_weight":5,
#     "gamma":0.1,
#     "max_depth": 4,
#     "eta": 0.01,
#     "silent": 1,
#     "subsample":0.75,
#     "colsample_bytree":0.55,
#     "scale_pos_weight":0.9
# }


params = {
    "objective": 'reg:linear',
    "eval_metric":'rmse',
    "seed":32,
    "booster": "gbtree",
    "min_child_weight":5,
    "gamma":0.1,
    "max_depth": 4,
    "eta": 0.009,
    "silent": 1,
    "subsample":0.75,
    "colsample_bytree":0.45,
    "scale_pos_weight":0.9
}


df_train = xgb.DMatrix(train, label)
gbm = xgb.train(params,df_train,num_boost_round=1000)

df_test = xgb.DMatrix(test)
y_pred = gbm.predict(df_test)

result = pd.DataFrame(id, columns=['Id'])
result['Pred'] = y_pred
result.to_csv('model/result_.csv',header=True,index=False)
# la = pd.DataFrame(label, columns=['label'])
# print(la.head(100))

print("cost time: " + str(time.time() - start))


