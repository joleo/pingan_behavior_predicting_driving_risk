# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_integrate2
   Description :
   Author :       Administrator
   date：          2018/5/21 0021
-------------------------------------------------
   Change Activity:
                   2018/5/21 0021:
-------------------------------------------------
"""
__author__ = 'Administrator'
from feature_extraction2 import *
fe = LYFeatureExtraction()

class FeatureIntegrate(object):

    def train_feature_integrate(self, data):
        user_hour_feature = fe.user_hour_feature(data)
        return user_hour_feature

    def test_feature_integrate(self, data):
        user_hour_feature = fe.user_hour_feature(data)
        return user_hour_feature

if __name__ == '__main__':
    # fi = FeatureIntegrate()
    # data = pd.read_csv('data/dm/train.csv')
    #
    # train_x = fi.train_feature_integrate(data)
    # feature = ['hour_0_kurt_height', 'hour_1_kurt_height','hour_2_kurt_height','hour_3_kurt_height','hour_4_kurt_height','hour_5_kurt_height','hour_6_kurt_height','hour_7_kurt_height','hour_8_kurt_height','hour_9_kurt_height','hour_10_kurt_height','hour_11_kurt_height','hour_12_kurt_height','hour_13_kurt_height','hour_14_kurt_height','hour_15_kurt_height','hour_16_kurt_height','hour_17_kurt_height','hour_18_kurt_height','hour_19_kurt_height','hour_20_kurt_height','hour_21_kurt_height','hour_22_kurt_height','hour_23_kurt_height']
    # train_x = train_x[feature]
    #
    # # uid = list(data['TERMINALNO'].unique())
    # # print(uid)
    # print(train_x.head(2).T)
    a = ['a','b']
    b = ['c', 'd']
    print(a+b)