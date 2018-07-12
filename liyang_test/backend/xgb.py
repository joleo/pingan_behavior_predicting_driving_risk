# -*- coding: utf-8 -*-
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

from ly_feature_integrate2 import *
lyfi = LYFeatureIntegrate2()
warnings.filterwarnings('ignore')

start = time.time()

all_features_list,user_Y_list = lyfi.get_all_features()
userid_list,test_features = lyfi.get_test_features02()

train = np.array(all_features_list)
label = np.array(user_Y_list).T
test = np.array(test_features)
id = np.array(userid_list).T


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


df_train = xgb.DMatrix(train, label)
gbm = xgb.train(params,df_train,num_boost_round=800)

df_test = xgb.DMatrix(test)
y_pred = gbm.predict(df_test)

result = pd.DataFrame(id, columns=['Id'])
result['Pred'] = y_pred
result.to_csv('model/result_.csv',header=True,index=False)

print("cost time: " + str(time.time() - start))


