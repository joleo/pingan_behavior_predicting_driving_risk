# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ly_feature_etl
   Description :
   Author :       Administrator
   date：          2018/5/8 0008
-------------------------------------------------
   Change Activity:
                   2018/5/8 0008:
-------------------------------------------------
"""
__author__ = 'Administrator'
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     LYFeatureExtraction2
   Description :
   Author :       Administrator
   date：          2018/5/7 0007
-------------------------------------------------
   Change Activity:
                   2018/5/7 0007:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd
import numpy as np
import datetime
import time
from math import radians, cos, sin, asin, sqrt
from config import *

class LYFeatureExtraction1(object):
    def __init__(self, d_h, data):
        self.userlist = d_h.get_userlist(data)

    def haversine1(self, lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # 地球平均半径，单位为公里
        return c * r * 1000

    def timestamp_datetime(self, value):
        fmt = '%Y-%m-%d %H:%M:%S'

        value = time.localtime(value)
        dt = time.strftime(fmt, value)
        return dt

    def user_Y(self, data):
        user_id = data.groupby(['TERMINALNO'], as_index = False)['Y'].first()
        return user_id

    def driver_time_feature(self, data):
        """
        时间特征
        """
        driver_time_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'LONGITUDE', 'HEIGHT', 'TIME']]
        driver_time_data['hour'] = driver_time_data['TIME'].map(lambda x: datetime.datetime.fromtimestamp(x).hour)
        driver_time_rate = driver_time_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].count()
        # 闲时
        driver_time_data.loc[:, 'is_hour_5'] = driver_time_data.hour.map(lambda x: 1 if x == 5 else 0)
        driver_time_data.loc[:, 'is_hour_6'] = driver_time_data.hour.map(lambda x: 1 if x == 6 else 0)
        driver_time_data.loc[:, 'is_hour_10'] = driver_time_data.hour.map(lambda x: 1 if x == 10 else 0)
        driver_time_data.loc[:, 'is_hour_15'] = driver_time_data.hour.map(lambda x: 1 if x == 15 else 0)
        driver_time_data.loc[:, 'is_hour_16'] = driver_time_data.hour.map(lambda x: 1 if x == 16 else 0)
        # 上下班高峰期
        driver_time_data.loc[:, 'is_hour_7'] = driver_time_data.hour.map(lambda x: 1 if x == 7 else 0)
        driver_time_data.loc[:, 'is_hour_8'] = driver_time_data.hour.map(lambda x: 1 if x == 8 else 0)
        driver_time_data.loc[:, 'is_hour_9'] = driver_time_data.hour.map(lambda x: 1 if x == 9 else 0)
        driver_time_data.loc[:, 'is_hour_17'] = driver_time_data.hour.map(lambda x: 1 if x == 17 else 0)
        driver_time_data.loc[:, 'is_hour_18'] = driver_time_data.hour.map(lambda x: 1 if x == 18 else 0)
        driver_time_data.loc[:, 'is_hour_19'] = driver_time_data.hour.map(lambda x: 1 if x == 19 else 0)
        driver_time_data.loc[:, 'is_hour_20'] = driver_time_data.hour.map(lambda x: 1 if x == 20 else 0)
        # 中午
        driver_time_data.loc[:, 'is_hour_11'] = driver_time_data.hour.map(lambda x: 1 if x == 11 else 0)
        driver_time_data.loc[:, 'is_hour_12'] = driver_time_data.hour.map(lambda x: 1 if x == 12 else 0)
        driver_time_data.loc[:, 'is_hour_13'] = driver_time_data.hour.map(lambda x: 1 if x == 13 else 0)
        driver_time_data.loc[:, 'is_hour_14'] = driver_time_data.hour.map(lambda x: 1 if x == 14 else 0)
        # 夜间
        driver_time_data.loc[:, 'is_hour_0'] = driver_time_data.hour.map(lambda x: 1 if x == 0 else 0)
        driver_time_data.loc[:, 'is_hour_1'] = driver_time_data.hour.map(lambda x: 1 if x == 1 else 0)
        driver_time_data.loc[:, 'is_hour_2'] = driver_time_data.hour.map(lambda x: 1 if x == 2 else 0)
        driver_time_data.loc[:, 'is_hour_3'] = driver_time_data.hour.map(lambda x: 1 if x == 3 else 0)
        driver_time_data.loc[:, 'is_hour_4'] = driver_time_data.hour.map(lambda x: 1 if x == 4 else 0)
        driver_time_data.loc[:, 'is_hour_21'] = driver_time_data.hour.map(lambda x: 1 if x == 21 else 0)
        driver_time_data.loc[:, 'is_hour_22'] = driver_time_data.hour.map(lambda x: 1 if x == 22 else 0)
        driver_time_data.loc[:, 'is_hour_23'] = driver_time_data.hour.map(lambda x: 1 if x == 23 else 0)

        driver_time_colmns = ['is_hour_5', 'is_hour_6', 'is_hour_10', 'is_hour_15', 'is_hour_16',
                              'is_hour_7', 'is_hour_8', 'is_hour_9', 'is_hour_17', 'is_hour_18',
                              'is_hour_19', 'is_hour_20', 'is_hour_11', 'is_hour_12', 'is_hour_13',
                              'is_hour_14'
                             , 'is_hour_0', 'is_hour_1', 'is_hour_2', 'is_hour_3',
                              'is_hour_4', 'is_hour_21', 'is_hour_22', 'is_hour_23'
                                ]

        # 时间占比特征
        driver_time_num = driver_time_data.groupby(['TERMINALNO'], as_index=False)[driver_time_colmns].sum()

        driver_time_rate.loc[:, 'is_hour_5_rate'] = driver_time_num['is_hour_5'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_6_rate'] = driver_time_num['is_hour_6'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_10_rate'] = driver_time_num['is_hour_10'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_15_rate'] = driver_time_num['is_hour_15'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_16_rate'] = driver_time_num['is_hour_16'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_7_rate'] = driver_time_num['is_hour_7'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_8_rate'] = driver_time_num['is_hour_8'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_9_rate'] = driver_time_num['is_hour_9'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_17_rate'] = driver_time_num['is_hour_17'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_18_rate'] = driver_time_num['is_hour_18'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_19_rate'] = driver_time_num['is_hour_19'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_20_rate'] = driver_time_num['is_hour_20'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_11_rate'] = driver_time_num['is_hour_11'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_12_rate'] = driver_time_num['is_hour_12'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_13_rate'] = driver_time_num['is_hour_13'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_14_rate'] = driver_time_num['is_hour_14'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_0_rate'] = driver_time_num['is_hour_0'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_1_rate'] = driver_time_num['is_hour_1'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_2_rate'] = driver_time_num['is_hour_2'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_3_rate'] = driver_time_num['is_hour_3'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_4_rate'] = driver_time_num['is_hour_4'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_21_rate'] = driver_time_num['is_hour_21'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_22_rate'] = driver_time_num['is_hour_22'] / driver_time_rate['TRIP_ID']
        driver_time_rate.loc[:, 'is_hour_23_rate'] = driver_time_num['is_hour_23'] / driver_time_rate['TRIP_ID']

        predictors1.append('is_hour_5_rate')
        predictors1.append('is_hour_6_rate')
        predictors1.append('is_hour_10_rate')
        predictors1.append('is_hour_15_rate')
        predictors1.append('is_hour_16_rate')
        predictors1.append('is_hour_7_rate')
        predictors1.append('is_hour_8_rate')
        predictors1.append('is_hour_9_rate')
        predictors1.append('is_hour_17_rate')
        predictors1.append('is_hour_18_rate')
        predictors1.append('is_hour_19_rate')
        predictors1.append('is_hour_20_rate')
        predictors1.append('is_hour_11_rate')
        predictors1.append('is_hour_12_rate')
        predictors1.append('is_hour_13_rate')
        predictors1.append('is_hour_14_rate')
        predictors1.append('is_hour_0_rate')
        predictors1.append('is_hour_1_rate')
        predictors1.append('is_hour_2_rate')
        predictors1.append('is_hour_3_rate')
        predictors1.append('is_hour_4_rate')
        predictors1.append('is_hour_21_rate')
        predictors1.append('is_hour_22_rate')
        predictors1.append('is_hour_23_rate')


        res = [
            driver_time_rate['is_hour_5_rate'].tolist(),
            driver_time_rate['is_hour_6_rate'].tolist(),
            driver_time_rate['is_hour_10_rate'].tolist(),
            driver_time_rate['is_hour_15_rate'].tolist(),
            driver_time_rate['is_hour_16_rate'].tolist(),
            driver_time_rate['is_hour_7_rate'].tolist(),
            driver_time_rate['is_hour_8_rate'].tolist(),
            driver_time_rate['is_hour_9_rate'].tolist(),
            driver_time_rate['is_hour_17_rate'].tolist(),
            driver_time_rate['is_hour_18_rate'].tolist(),
            driver_time_rate['is_hour_19_rate'].tolist(),
            driver_time_rate['is_hour_20_rate'].tolist(),
            driver_time_rate['is_hour_11_rate'].tolist(),
            driver_time_rate['is_hour_12_rate'].tolist(),
            driver_time_rate['is_hour_13_rate'].tolist(),
            driver_time_rate['is_hour_14_rate'].tolist(),
            driver_time_rate['is_hour_0_rate'].tolist(),
            driver_time_rate['is_hour_1_rate'].tolist(),
            driver_time_rate['is_hour_2_rate'].tolist(),
            driver_time_rate['is_hour_3_rate'].tolist(),
            driver_time_rate['is_hour_4_rate'].tolist(),
            driver_time_rate['is_hour_21_rate'].tolist(),
            driver_time_rate['is_hour_22_rate'].tolist(),
            driver_time_rate['is_hour_23_rate'].tolist()
        ]
        return res

    def get_feat(self, train):
        """
        夜晚开车和每天用车时长的统计
        """
        train['TIME'] = train['TIME'].apply(lambda x: self.timestamp_datetime(x), 1)
        train['TIME'] = train['TIME'].apply(lambda x: str(x)[:13], 1)
        train = train.sort_values(by=["TERMINALNO", 'TIME'])
        train.index = range(len(train))
        train['hour'] = train.TIME.apply(lambda x: str(x)[11:13], 1)
        train['hour'] = train['hour'].astype(int)

        train['is_hour_0'] = train.hour.apply(lambda x: 1 if x == 0 else 0, 1)
        train['is_hour_1'] = train.hour.apply(lambda x: 1 if x == 1 else 0, 1)
        train['is_hour_2'] = train.hour.apply(lambda x: 1 if x == 2 else 0, 1)
        train['is_hour_3'] = train.hour.apply(lambda x: 1 if x == 3 else 0, 1)
        train['is_hour_4'] = train.hour.apply(lambda x: 1 if x == 4 else 0, 1)
        train['is_hour_21'] = train.hour.apply(lambda x: 1 if x == 21 else 0, 1)
        train['is_hour_22'] = train.hour.apply(lambda x: 1 if x == 22 else 0, 1)
        train['is_hour_23'] = train.hour.apply(lambda x: 1 if x == 23 else 0, 1)

        train_hour = train.groupby(['TERMINALNO', 'TIME'], as_index=False).count()
        train_hour.TIME = train_hour.TIME.apply(lambda x: str(x)[:10], 1)
        train_day = train_hour.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).count()

        train_hour_count = train_day.groupby('TERMINALNO')['LONGITUDE'].agg(
            {"hour_count_max": "max", "hour_count_min": "min", "hour_count_mean": "mean", "hour_count_std": "std",
             "hour_count_skew": "skew"}).reset_index()

        train_hour_first = train.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).first()
        train_hour_first.TIME = train_hour_first.TIME.apply(
            lambda x: str(x)[:10], 1)
        train_day_sum = train_hour_first.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).sum()
        train_day_sum['hour_count'] = train_day['LONGITUDE']

        train_day_sum['night_drive_count'] = train_day_sum.apply(lambda x: x['is_hour_0'] + x['is_hour_1'] +
                                                                           x['is_hour_2'] + x['is_hour_3'] + x[
                                                                               'is_hour_4'] +
                                                                           x['is_hour_21'] + x['is_hour_22'] + x[
                                                                               'is_hour_23'], 1)

        train_day_sum['night_delta'] = train_day_sum['night_drive_count'] / \
                                       train_day_sum['hour_count']

        train_day_sum['is_night'] = train_day_sum['night_drive_count'].apply(
            lambda x: 1 if x != 0 else 0, 1)
        train_hour_count['night__day_delta'] = train_day_sum.groupby(['TERMINALNO'], as_index=False).sum(
        )['is_night'] / (train_day_sum.groupby(['TERMINALNO'], as_index=False).count()['HEIGHT'])

        train_night_count = train_day_sum.groupby('TERMINALNO')['night_delta'].agg(
            {"night_count_max": "max", "night_count_min": "min", "night_count_mean": "mean", "night_count_std": "std",
             "night_count_skew": "skew"}).reset_index()

        train_data = pd.merge(
            train_hour_count, train_night_count, on="TERMINALNO", how="left")

        predictors1.append('hour_count_max')
        predictors1.append('hour_count_min')
        predictors1.append('hour_count_mean')
        predictors1.append('hour_count_std')
        predictors1.append('hour_count_skew')
        predictors1.append('night__day_delta')
        predictors1.append('night_count_max')
        predictors1.append('night_count_min')
        predictors1.append('night_count_mean')
        predictors1.append('night_count_std')
        predictors1.append('night_count_skew')

        res = [train_data['hour_count_max'].tolist(),
               train_data['hour_count_min'].tolist(),
               train_data['hour_count_mean'].tolist(),
               train_data['hour_count_std'].tolist(),
               train_data['hour_count_skew'].tolist(),
               train_data['night__day_delta'].tolist(),
               train_data['night_count_max'].tolist(),
               train_data['night_count_min'].tolist(),
               train_data['night_count_mean'].tolist(),
               train_data['night_count_std'].tolist(),
               train_data['night_count_skew'].tolist()
               ]

        return res

    def driver_base_feature(self, data):
        # trid 特征
        df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()

        # 驾驶行为特征
        user_speed_feature = data.groupby(['TERMINALNO'])['SPEED'].agg(['mean','std','max']).reset_index()
        user_height_feature = data.groupby(['TERMINALNO'])['HEIGHT'].agg(['mean','std']).reset_index()
        user_direction_feature = data.groupby(['TERMINALNO'])['DIRECTION'].agg(['mean']).reset_index()

        # 合并
        driver_base_feature = pd.merge(df_user_trid_num, user_speed_feature, how='left', on='TERMINALNO')
        driver_base_feature = pd.merge(driver_base_feature, user_height_feature, how='left', on='TERMINALNO')
        driver_base_feature = pd.merge(driver_base_feature, user_direction_feature, how='left', on='TERMINALNO')

        driver_base_feature.rename(columns={'mean_x': 'mean_speed', 'std_x': 'var_speed', 'max':'max_speed',
                                            'mean_y': 'mean_height', 'std_y': 'var_height',
                                            'mean': 'mean_direction'}, inplace=True)

        predictors1.append('TRIP_ID')
        predictors1.append('mean_speed')
        predictors1.append('var_speed')
        predictors1.append('max_speed')
        predictors1.append('mean_height')
        predictors1.append('var_height')
        predictors1.append('mean_direction')

        res = [driver_base_feature['TRIP_ID'].tolist(),
               driver_base_feature['mean_speed'].tolist(),
               driver_base_feature['var_speed'].tolist(),
               driver_base_feature['max_speed'].tolist(),
               driver_base_feature['mean_height'].tolist(),
               driver_base_feature['var_height'].tolist(),
               driver_base_feature['mean_direction'].tolist()
               ]

        return res

    def user_callstate_feature(self, data):
        # 通话状态特征
        callstate_data = data.loc[:, ['TERMINALNO', 'CALLSTATE']]
        user_callstate_rate = callstate_data.groupby(['TERMINALNO'], as_index=False)['CALLSTATE'].count()

        # 用户通话特征
        # 未知状态占比
        callstate_data['call_unknow_state'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 0 else 0)
        user_callstate_rate.loc[:, 'call_unknow_state'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'call_unknow_state'].sum()
        user_callstate_rate.loc[:, 'call_unknow_state_rate'] = user_callstate_rate['call_unknow_state'] / \
                                                               user_callstate_rate['CALLSTATE']
        # 用户呼入占比
        callstate_data['user_call_in_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 2 else 0)
        user_callstate_rate.loc[:, 'user_call_in_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_in_num'].sum()
        user_callstate_rate.loc[:, 'user_call_in_rate'] = user_callstate_rate['user_call_in_num'] / user_callstate_rate[
            'CALLSTATE']

        # 用户呼出占比
        callstate_data['user_call_out_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 1 else 0)
        user_callstate_rate.loc[:, 'user_call_out_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_out_num'].sum()
        user_callstate_rate.loc[:, 'user_call_out_rate'] = user_callstate_rate['user_call_out_num'] / \
                                                           user_callstate_rate['CALLSTATE']

        # 用户连通占比
        callstate_data['user_call_connection_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 3 else 0)
        user_callstate_rate.loc[:, 'user_call_connection_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_connection_num'].sum()
        user_callstate_rate.loc[:, 'user_call_connection_rate'] = user_callstate_rate['user_call_connection_num'] / \
                                                                  user_callstate_rate['CALLSTATE']

        # 用户断连占比
        callstate_data['user_call_close_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 4 else 0)
        user_callstate_rate.loc[:, 'user_call_close_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_close_num'].sum()
        user_callstate_rate.loc[:, 'user_call_close_rate'] = user_callstate_rate['user_call_close_num'] / \
                                                             user_callstate_rate['CALLSTATE']

        """
        到某一点的距离
        """
        lon_data = data.loc[:, ['TERMINALNO', 'LONGITUDE', 'LATITUDE']]
        lon_first_data = lon_data.groupby('TERMINALNO', as_index=False)[['LONGITUDE', 'LATITUDE']].first()
        lon_first_data.loc[:, 'to_dis'] = lon_first_data.apply(
            lambda row: self.haversine1(row['LONGITUDE'], row['LATITUDE'], 113.9177317, 22.54334333), axis=1)

        predictors1.append('call_unknow_state_rate')
        predictors1.append('user_call_in_rate')
        predictors1.append('user_call_out_rate')
        predictors1.append('user_call_connection_rate')
        predictors1.append('user_call_close_rate')
        predictors1.append('to_dis')

        res = [user_callstate_rate['call_unknow_state_rate'].tolist(),
               user_callstate_rate['user_call_in_rate'].tolist(),
               user_callstate_rate['user_call_out_rate'].tolist(),
               user_callstate_rate['user_call_connection_rate'].tolist(),
               user_callstate_rate['user_call_close_rate'].tolist(),
               lon_first_data['to_dis'].tolist()
               ]

        return res


    def user_direction__rate(self, data):
        """
        用户方向占比
        """
        direction_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'DIRECTION']]
        user_direction_num = direction_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].count()
        direction_data['user_direction_30_num'] = direction_data['DIRECTION'].map(lambda x: 1 if x < 30 else 0)
        user_direction_num.loc[:, 'user_direction_30_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_30_num'].sum()
        user_direction_num.loc[:, 'user_direction_30_rate'] = user_direction_num['user_direction_30_num'] / \
                                                              user_direction_num['TRIP_ID']
        direction_data['user_direction_30_90_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 30 and x < 90 else 0)
        user_direction_num.loc[:, 'user_direction_30_90_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_30_90_num'].sum()
        user_direction_num.loc[:, 'user_direction_30_90_rate'] = user_direction_num['user_direction_30_90_num'] / \
                                                                 user_direction_num['TRIP_ID']
        direction_data['user_direction_90_120_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 90 and x < 120 else 0)
        user_direction_num.loc[:, 'user_direction_90_120_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_90_120_num'].sum()
        user_direction_num.loc[:, 'user_direction_90_120_rate'] = user_direction_num['user_direction_90_120_num'] / \
                                                                  user_direction_num['TRIP_ID']
        direction_data['user_direction_120_180_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 120 and x < 180 else 0)
        user_direction_num.loc[:, 'user_direction_120_180_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_120_180_num'].sum()
        user_direction_num.loc[:, 'user_direction_120_180_rate'] = user_direction_num['user_direction_120_180_num'] / \
                                                                   user_direction_num['TRIP_ID']
        direction_data['user_direction_180_210_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 180 and x < 210 else 0)
        user_direction_num.loc[:, 'user_direction_180_210_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_180_210_num'].sum()
        user_direction_num.loc[:, 'user_direction_180_210_rate'] = user_direction_num['user_direction_180_210_num'] / \
                                                                   user_direction_num[
                                                                       'TRIP_ID']
        direction_data['user_direction__210_360_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 210 and x < 360 else 0)
        user_direction_num.loc[:, 'user_direction__210_360_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction__210_360_num'].sum()
        user_direction_num.loc[:, 'user_direction__210_360_rate'] = user_direction_num['user_direction__210_360_num'] / \
                                                                    user_direction_num[
                                                                        'TRIP_ID']

        predictors1.append('user_direction_30_rate')
        predictors1.append('user_direction_30_90_rate')
        predictors1.append('user_direction_90_120_rate')
        predictors1.append('user_direction_120_180_rate')
        predictors1.append('user_direction_180_210_rate')
        predictors1.append('user_direction__210_360_rate')

        res = [user_direction_num['user_direction_30_rate'].tolist(),
               user_direction_num['user_direction_30_90_rate'].tolist(),
               user_direction_num['user_direction_90_120_rate'].tolist(),
               user_direction_num['user_direction_120_180_rate'].tolist(),
               user_direction_num['user_direction_180_210_rate'].tolist(),
               user_direction_num['user_direction__210_360_rate'].tolist()
               ]
        return res

    def user_height_rate(self, data):
        """
         海拔占比
        """
        height_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'HEIGHT']]
        user_height_num = height_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()
        user_height_std = height_data.groupby(['TERMINALNO', 'TRIP_ID'])['HEIGHT'].agg(['std']).reset_index()

        user_height_std['user_height_std'] = user_height_std['std'].replace(np.nan, -100)
        user_height_std.loc[:, 'user_height_nan_num'] = user_height_std['user_height_std'].map(
            lambda x: 1 if x == -100 else 0)
        user_height_num.loc[:, 'user_height_nan_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_nan_num'].sum()
        user_height_num.loc[:, 'user_height_nan_rate'] = user_height_num['user_height_nan_num'] / user_height_num[
            'TRIP_ID']
        user_height_std.loc[:, 'user_height_2_num'] = user_height_std['std'].map(lambda x: 1 if x < 2 else 0)
        user_height_num.loc[:, 'user_height_2_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_2_num'].sum()
        user_height_num.loc[:, 'user_height_std_2_rate'] = user_height_num['user_height_2_num'] / user_height_num[
            'TRIP_ID']
        user_height_std.loc[:, 'user_height_2_5_num'] = user_height_std['std'].map(
            lambda x: 1 if x > 2 and x < 5 else 0)
        user_height_num.loc[:, 'user_height_2_5_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_2_5_num'].sum()
        user_height_num.loc[:, 'user_height_std_2_5_rate'] = user_height_num['user_height_2_5_num'] / user_height_num[
            'TRIP_ID']
        user_height_std.loc[:, 'user_height_5_10_num'] = user_height_std['std'].map(
            lambda x: 1 if x >= 5 and x < 10 else 0)
        user_height_num.loc[:, 'user_height_5_10_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_5_10_num'].sum()
        user_height_num.loc[:, 'user_height_std_5_10_rate'] = user_height_num['user_height_5_10_num'] / \
                                                              user_height_num[
                                                                  'TRIP_ID']
        user_height_std.loc[:, 'user_height_10_num'] = user_height_std['std'].map(
            lambda x: 1 if x >= 10 else 0)
        user_height_num.loc[:, 'user_height_10_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_10_num'].sum()
        user_height_num.loc[:, 'user_height_std_10_rate'] = user_height_num['user_height_10_num'] / \
                                                            user_height_num[
                                                                'TRIP_ID']

        user_height_avg_height = height_data.groupby(['TERMINALNO', 'TRIP_ID'], as_index=False)['HEIGHT'].mean()

        user_height_avg_height.loc[:, 'user_low_height_num'] = user_height_avg_height['HEIGHT'].map(
            lambda x: 1 if x < 100 else 0)
        user_height_num.loc[:, 'user_low_height_num'] = user_height_avg_height.groupby(['TERMINALNO'], as_index=False)[
            'user_low_height_num'].sum()
        user_height_num.loc[:, 'user_low_height__rate'] = user_height_num['user_low_height_num'] / user_height_num[
            'TRIP_ID']

        user_height_avg_height.loc[:, 'user_middle_height_num'] = user_height_avg_height['HEIGHT'].map(
            lambda x: 1 if x > 100 and x < 1500 else 0)
        user_height_num.loc[:, 'user_middle_height_num'] = \
        user_height_avg_height.groupby(['TERMINALNO'], as_index=False)[
            'user_middle_height_num'].sum()
        user_height_num.loc[:, 'user_middle_height__rate'] = user_height_num['user_middle_height_num'] / \
                                                             user_height_num[
                                                                 'TRIP_ID']
        user_height_avg_height.loc[:, 'user_high_height_num'] = user_height_avg_height['HEIGHT'].map(
            lambda x: 1 if x > 1500 and x < 3500 else 0)
        user_height_num.loc[:, 'user_high_height_num'] = user_height_avg_height.groupby(['TERMINALNO'], as_index=False)[
            'user_high_height_num'].sum()
        user_height_num.loc[:, 'user_high_height_rate'] = user_height_num['user_high_height_num'] / user_height_num[
            'TRIP_ID']

        predictors1.append('user_height_nan_rate')
        predictors1.append('user_height_std_2_rate')
        predictors1.append('user_height_std_2_5_rate')
        predictors1.append('user_height_std_5_10_rate')
        predictors1.append('user_height_std_10_rate')
        predictors1.append('user_low_height__rate')
        predictors1.append('user_middle_height__rate')
        predictors1.append('user_high_height_rate')


        res = [user_height_num['user_height_nan_rate'].tolist(),
               user_height_num['user_height_std_2_rate'].tolist(),
               user_height_num['user_height_std_2_5_rate'].tolist(),
               user_height_num['user_height_std_5_10_rate'].tolist(),
               user_height_num['user_height_std_10_rate'].tolist(),
               user_height_num['user_low_height__rate'].tolist(),
               user_height_num['user_middle_height__rate'].tolist(),
               user_height_num['user_high_height_rate'].tolist()

               ]
        return res

    def user_speed_rate(self, data):
        """
        速度占比
        """
        speed_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'SPEED']]
        user_speed_num = speed_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()
        user_speed_std = speed_data.groupby(['TERMINALNO', 'TRIP_ID'])['SPEED'].agg(['std']).reset_index()
        # 速度方差最大的行程
        user_speed_std_max = user_speed_std.groupby(['TERMINALNO'], as_index=False)['std'].max()

        # speed 方差为空占比
        user_speed_std['user_speed_std'] = user_speed_std['std'].replace(np.nan, 99999)
        user_speed_std.loc[:, 'user_rapid_std_nan_speed'] = user_speed_std['user_speed_std'].map(
            lambda x: 1 if x == 99999 else 0)
        user_speed_num.loc[:, 'user_rapid_std_nan_speed'] = user_speed_std.groupby(['TERMINALNO'], as_index=False)[
            'user_rapid_std_nan_speed'].sum()
        user_speed_num.loc[:, 'user_speed_nan_rate'] = user_speed_num['user_rapid_std_nan_speed'] / user_speed_num[
            'TRIP_ID']

        # # 速度方差小于2，说明速度平稳
        user_speed_std.loc[:, 'user_speed_2_std'] = user_speed_std['user_speed_std'].map(lambda x: 1 if x < 2 else 0)
        user_speed_num.loc[:, 'user_speed_2_std'] = user_speed_std.groupby(['TERMINALNO'], as_index=False)[
            'user_speed_2_std'].sum()
        user_speed_num.loc[:, 'user_speed_2_rate'] = user_speed_num['user_speed_2_std'] / user_speed_num[
            'TRIP_ID']

        # 速度方差大于2且小于10，说明速度变动
        user_speed_std.loc[:, 'user_speed_2_10_std'] = user_speed_std['user_speed_std'].map(
            lambda x: 1 if x > 2 and x < 10 else 0)
        user_speed_num.loc[:, 'user_speed_2_10_std'] = user_speed_std.groupby(['TERMINALNO'], as_index=False)[
            'user_speed_2_10_std'].sum()
        user_speed_num.loc[:, 'user_speed_2_10_rate'] = user_speed_num['user_speed_2_10_std'] / user_speed_num[
            'TRIP_ID']
        # 方差大于10的行程作为急加/减速
        user_speed_std.loc[:, 'user_speed_10_std'] = user_speed_std['user_speed_std'].map(lambda x: 1 if x > 10 else 0)
        user_speed_num.loc[:, 'user_speed_10_std'] = user_speed_std.groupby(['TERMINALNO'], as_index=False)[
            'user_speed_10_std'].sum()
        user_speed_num.loc[:, 'user_speed_10_rate'] = user_speed_num['user_speed_10_std'] / user_speed_num[
            'TRIP_ID']
        user_speed_num = pd.merge(user_speed_std_max, user_speed_num, how='left', on='TERMINALNO')

        user_speed_num.rename(columns={'std': 'user_rapid_speed_max_std'}, inplace=True)



        predictors1.append('user_rapid_speed_max_std')
        predictors1.append('user_speed_nan_rate')
        predictors1.append('user_speed_2_rate')
        predictors1.append('user_speed_2_10_rate')
        predictors1.append('user_speed_10_rate')

        res = [user_speed_num['user_rapid_speed_max_std'].tolist(),
               user_speed_num['user_speed_nan_rate'].tolist(),
               user_speed_num['user_speed_2_rate'].tolist(),
               user_speed_num['user_speed_2_10_rate'].tolist(),
               user_speed_num['user_speed_10_rate'].tolist(),

               ]
        return res