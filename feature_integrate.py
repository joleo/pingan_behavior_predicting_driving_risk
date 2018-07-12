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

from feature_extraction import *
fe = LYFeatureExtraction()

class FeatureIntegrate(object):

    def train_feature_integrate(self, data):

        user_driver_stat = fe.user_driver_stat(data) # 用户驾驶行为基本特征统计
        user_driver_time = fe.user_driver_time(data) # 用户时间特征特征
        user_night_stat = fe.user_night_stat(data) # 夜晚开车和每天用车时长的统计
        get_distance = fe.get_distance(data) # 经纬度转距离
        get_user_Y = fe.get_user_Y(data) # 获取用户Y值

        # user_geo_stat = fe.user_geo_stat(data)

        feature = pd.merge(user_driver_stat, user_driver_time, how='left', on='TERMINALNO')
        feature = pd.merge(feature, user_night_stat, how='left', on='TERMINALNO')
        feature = pd.merge(feature, get_distance, how='left', on='TERMINALNO')
        feature = pd.merge(feature, get_user_Y, how='left', on='TERMINALNO')
        # feature = pd.merge(feature, user_geo_stat, how='left', on='TERMINALNO')


        return feature

    def test_feature_integrate(self, data):
        driver_test_base_feature = fe.user_test_driver_stat(data)  # 驾驶行为基本特征
        user_driver_time = fe.user_driver_time(data)  # 用户时间特征特征
        user_night_stat = fe.user_night_stat(data)  # 夜晚开车和每天用车时长的统计
        get_distance = fe.get_distance(data)  # 经纬度转距离
        # user_geo_stat = fe.user_geo_stat(data)

        feature = pd.merge(driver_test_base_feature, user_driver_time, how='left', on='TERMINALNO')
        feature = pd.merge(feature, user_night_stat, how='left', on='TERMINALNO')
        feature = pd.merge(feature, get_distance, how='left', on='TERMINALNO')
        # feature = pd.merge(feature, user_geo_stat, how='left', on='TERMINALNO')

        return feature

if __name__ == '__main__':
    fi = FeatureIntegrate()
    data = pd.read_csv('data/dm/train.csv')
    fea = fi.train_feature_integrate(data)
    # data['TIME'] = data['TIME'].map(lambda x: datetime.datetime.fromtimestamp(x))
    # data.to_csv('model/train_time.csv')
    print(fea.info())



