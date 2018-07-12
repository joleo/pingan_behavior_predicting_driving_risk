# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ly_feature_etl2
   Description :
   Author :       Administrator
   date：          2018/5/8 0008
-------------------------------------------------
   Change Activity:
                   2018/5/8 0008:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd
import numpy as np
import datetime
import time
from math import radians, cos, sin, asin, sqrt
from config import *

class LYFeatureExtraction2(object):

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

    def get_user_Y(self, data):
        """
        获取用户Y值
        :param data:
        :return:
        """
        user_id = data.groupby(['TERMINALNO'], as_index = False)['Y'].first()
        return user_id

    def user_driver_time(self, data):
        """
        用户时间特征特征
        """
        user_time_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'TIME']]
        user_time_data['hour'] = user_time_data['TIME'].map(lambda x: datetime.datetime.fromtimestamp(x).hour)
        user_time_rate = user_time_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].count()

        user_time_data.loc[:, 'hour_5'] = user_time_data.hour.map(lambda x: 1 if x == 5 else 0)
        user_time_data.loc[:, 'hour_6'] = user_time_data.hour.map(lambda x: 1 if x == 6 else 0)
        user_time_data.loc[:, 'hour_10'] = user_time_data.hour.map(lambda x: 1 if x == 10 else 0)
        user_time_data.loc[:, 'hour_15'] = user_time_data.hour.map(lambda x: 1 if x == 15 else 0)
        user_time_data.loc[:, 'hour_7'] = user_time_data.hour.map(lambda x: 1 if x == 7 else 0)
        user_time_data.loc[:, 'hour_8'] = user_time_data.hour.map(lambda x: 1 if x == 8 else 0)
        user_time_data.loc[:, 'hour_9'] = user_time_data.hour.map(lambda x: 1 if x == 9 else 0)
        user_time_data.loc[:, 'hour_17'] = user_time_data.hour.map(lambda x: 1 if x == 17 else 0)
        user_time_data.loc[:, 'hour_18'] = user_time_data.hour.map(lambda x: 1 if x == 18 else 0)
        user_time_data.loc[:, 'hour_11'] = user_time_data.hour.map(lambda x: 1 if x == 11 else 0)
        user_time_data.loc[:, 'hour_12'] = user_time_data.hour.map(lambda x: 1 if x == 12 else 0)
        user_time_data.loc[:, 'hour_13'] = user_time_data.hour.map(lambda x: 1 if x == 13 else 0)
        user_time_data.loc[:, 'hour_14'] = user_time_data.hour.map(lambda x: 1 if x == 14 else 0)
        user_time_data.loc[:, 'hour_0'] = user_time_data.hour.map(lambda x: 1 if x == 0 else 0)
        user_time_data.loc[:, 'hour_1'] = user_time_data.hour.map(lambda x: 1 if x == 1 else 0)
        user_time_data.loc[:, 'hour_2'] = user_time_data.hour.map(lambda x: 1 if x == 2 else 0)
        user_time_data.loc[:, 'hour_3'] = user_time_data.hour.map(lambda x: 1 if x == 3 else 0)
        user_time_data.loc[:, 'hour_4'] = user_time_data.hour.map(lambda x: 1 if x == 4 else 0)
        user_time_data.loc[:, 'hour_23'] = user_time_data.hour.map(lambda x: 1 if x == 23 else 0)

        driver_time_colmns = ['hour_5', 'hour_6', 'hour_10', 'hour_15',
                              'hour_7', 'hour_8', 'hour_9', 'hour_17', 'hour_18',
                              'hour_11', 'hour_12', 'hour_13',
                              'hour_14', 'hour_0', 'hour_1', 'hour_2', 'hour_3',
                              'hour_4', 'hour_23']

        # 计算每个时间段的比例
        driver_time_num = user_time_data.groupby(['TERMINALNO'], as_index=False)[driver_time_colmns].sum()

        user_time_rate.loc[:, 'hour_5_rate'] = driver_time_num['hour_5'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_6_rate'] = driver_time_num['hour_6'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_10_rate'] = driver_time_num['hour_10'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_15_rate'] = driver_time_num['hour_15'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_7_rate'] = driver_time_num['hour_7'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_8_rate'] = driver_time_num['hour_8'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_9_rate'] = driver_time_num['hour_9'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_17_rate'] = driver_time_num['hour_17'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_18_rate'] = driver_time_num['hour_18'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_11_rate'] = driver_time_num['hour_11'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_12_rate'] = driver_time_num['hour_12'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_13_rate'] = driver_time_num['hour_13'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_14_rate'] = driver_time_num['hour_14'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_0_rate'] = driver_time_num['hour_0'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_1_rate'] = driver_time_num['hour_1'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_2_rate'] = driver_time_num['hour_2'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_3_rate'] = driver_time_num['hour_3'] / user_time_rate['TRIP_ID']
        user_time_rate.loc[:, 'hour_4_rate'] = driver_time_num['hour_4'] / user_time_rate['TRIP_ID']
        # user_time_rate.loc[:, 'hour_23_rate'] = driver_time_num['hour_23'] / user_time_rate['TRIP_ID']


        predictors1.append('hour_5_rate')
        predictors1.append('hour_6_rate')
        predictors1.append('hour_10_rate')
        predictors1.append('hour_15_rate')
        predictors1.append('hour_7_rate')
        predictors1.append('hour_8_rate')
        predictors1.append('hour_9_rate')
        predictors1.append('hour_17_rate')
        predictors1.append('hour_18_rate')
        predictors1.append('hour_11_rate')
        predictors1.append('hour_12_rate')
        predictors1.append('hour_13_rate')
        predictors1.append('hour_14_rate')
        predictors1.append('hour_0_rate')
        predictors1.append('hour_1_rate')
        predictors1.append('hour_2_rate')
        predictors1.append('hour_3_rate')
        predictors1.append('hour_4_rate')
        predictors1.append('hour_23_rate')

        res = [
                user_time_rate['hour_5_rate'].tolist(),
                user_time_rate['hour_6_rate'].tolist(),
                user_time_rate['hour_10_rate'].tolist(),
                user_time_rate['hour_15_rate'].tolist(),
                user_time_rate['hour_7_rate'].tolist(),
                user_time_rate['hour_8_rate'].tolist(),
                user_time_rate['hour_9_rate'].tolist(),
                user_time_rate['hour_17_rate'].tolist(),
                user_time_rate['hour_18_rate'].tolist(),
                user_time_rate['hour_11_rate'].tolist(),
                user_time_rate['hour_12_rate'].tolist(),
                user_time_rate['hour_13_rate'].tolist(),
                user_time_rate['hour_14_rate'].tolist(),
                user_time_rate['hour_0_rate'].tolist(),
                user_time_rate['hour_1_rate'].tolist(),
                user_time_rate['hour_2_rate'].tolist(),
                user_time_rate['hour_3_rate'].tolist(),
                user_time_rate['hour_4_rate'].tolist()
                # ,user_time_rate['hour_23_rate'].tolist()
                ]

        return res

    def user_night_stat(self, train):
        """
        夜晚开车和每天用车时长的统计
        """
        train['TIME'] = train['TIME'].apply(lambda x: self.timestamp_datetime(x), 1)
        train['TIME'] = train['TIME'].apply(lambda x: str(x)[:13], 1)
        train = train.sort_values(by=["TERMINALNO", 'TIME'])
        train.index = range(len(train))
        train['hour'] = train.TIME.apply(lambda x: str(x)[11:13], 1)
        train['hour'] = train['hour'].astype(int)

        train['hour_0'] = train.hour.apply(lambda x: 1 if x == 0 else 0, 1)
        train['hour_1'] = train.hour.apply(lambda x: 1 if x == 1 else 0, 1)
        train['hour_2'] = train.hour.apply(lambda x: 1 if x == 2 else 0, 1)
        train['hour_3'] = train.hour.apply(lambda x: 1 if x == 3 else 0, 1)
        train['hour_4'] = train.hour.apply(lambda x: 1 if x == 4 else 0, 1)
        train['hour_21'] = train.hour.apply(lambda x: 1 if x == 21 else 0, 1)
        train['hour_22'] = train.hour.apply(lambda x: 1 if x == 22 else 0, 1)
        train['hour_23'] = train.hour.apply(lambda x: 1 if x == 23 else 0, 1)

        train_hour = train.groupby(['TERMINALNO', 'TIME'], as_index=False).count()
        train_hour.TIME = train_hour.TIME.apply(lambda x: str(x)[:10], 1)
        train_day = train_hour.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).count()

        train_hour_count = train_day.groupby('TERMINALNO')['LONGITUDE'].agg(
            {"hour_count_max": "max",  "hour_count_mean": "mean", "hour_count_std": "std",
             "hour_count_skew": "skew"}).reset_index()

        train_hour_first = train.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).first()
        train_hour_first.TIME = train_hour_first.TIME.apply(
            lambda x: str(x)[:10], 1)
        train_day_sum = train_hour_first.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).sum()
        train_day_sum['hour_count'] = train_day['LONGITUDE']

        train_day_sum['night_drive_count'] = train_day_sum.apply(lambda x: x['hour_0'] + x['hour_1'] +
                                                                           x['hour_2'] + x['hour_3'] + x[
                                                                               'hour_4'] +
                                                                           x['hour_21'] + x['hour_22'] + x[
                                                                               'hour_23'], 1)

        train_day_sum['night_delta'] = train_day_sum['night_drive_count'] / \
                                       train_day_sum['hour_count']

        train_day_sum['night'] = train_day_sum['night_drive_count'].apply(
            lambda x: 1 if x != 0 else 0, 1)
        train_hour_count['night__day_delta'] = train_day_sum.groupby(['TERMINALNO'], as_index=False).sum(
        )['night'] / (train_day_sum.groupby(['TERMINALNO'], as_index=False).count()['HEIGHT'])

        train_night_count = train_day_sum.groupby('TERMINALNO')['night_delta'].agg(
            {"night_count_max": "max",  "night_count_mean": "mean", "night_count_std": "std",
             "night_count_skew": "skew"}).reset_index()

        train_data = pd.merge(
            train_hour_count, train_night_count, on="TERMINALNO", how="left")

        predictors1.append('hour_count_max')
        predictors1.append('hour_count_mean')
        predictors1.append('hour_count_std')
        predictors1.append('hour_count_skew')
        predictors1.append('night__day_delta')
        predictors1.append('night_count_max')
        predictors1.append('night_count_mean')
        predictors1.append('night_count_std')
        predictors1.append('night_count_skew')

        res = [train_data['hour_count_max'].tolist(),
               train_data['hour_count_mean'].tolist(),
               train_data['hour_count_std'].tolist(),
               train_data['hour_count_skew'].tolist(),
               train_data['night__day_delta'].tolist(),
               train_data['night_count_max'].tolist(),
               train_data['night_count_mean'].tolist(),
               train_data['night_count_std'].tolist(),
               train_data['night_count_skew'].tolist()
               ]

        return res

    def user_driver_stat(self, data):
        """
        用户驾驶行为基本特征统计
        """
        # trid 特征
        df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()

        # 驾驶行为特征
        user_speed_stat = data.groupby(['TERMINALNO'])['SPEED'].agg(['mean','std','max']).reset_index()
        user_height_stat = data.groupby(['TERMINALNO'])['HEIGHT'].agg(['mean','std']).reset_index()
        user_direction_stat = data.groupby(['TERMINALNO'])['DIRECTION'].agg(['mean']).reset_index()

        # 合并
        user_driver_stat = pd.merge(df_user_trid_num, user_speed_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_height_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_direction_stat, how='left', on='TERMINALNO')

        user_driver_stat.rename(columns={'mean_x': 'mean_speed', 'std_x': 'var_speed', 'max':'max_speed',
                                            'mean_y': 'mean_height', 'std_y': 'var_height',
                                            'mean': 'mean_direction'}, inplace=True)

        predictors1.append('TRIP_ID')
        predictors1.append('mean_speed')
        predictors1.append('var_speed')
        predictors1.append('max_speed')
        predictors1.append('mean_height')
        predictors1.append('var_height')
        predictors1.append('mean_direction')

        res = [ user_driver_stat['TRIP_ID'].tolist(),
                user_driver_stat['mean_speed'].tolist(),
                user_driver_stat['var_speed'].tolist(),
                user_driver_stat['max_speed'].tolist(),
                user_driver_stat['mean_height'].tolist(),
                user_driver_stat['var_height'].tolist(),
                user_driver_stat['mean_direction'].tolist()
        ]

        return res

    # def user_test_driver_stat(self, data):
    #     """
    #     用户驾驶行为基本特征统计
    #     """
    #     # trid 特征
    #     df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()
    #
    #     # 驾驶行为特征
    #     user_speed_stat = data.groupby(['TERMINALNO'])['SPEED'].agg(['mean','std','max']).reset_index()
    #     user_height_stat = data.groupby(['TERMINALNO'])['HEIGHT'].agg(['mean','std']).reset_index()
    #     user_direction_stat = data.groupby(['TERMINALNO'])['DIRECTION'].agg(['mean']).reset_index()
    #
    #     # 合并
    #     user_driver_stat = pd.merge(df_user_trid_num, user_speed_stat, how='left', on='TERMINALNO')
    #     user_driver_stat = pd.merge(user_driver_stat, user_height_stat, how='left', on='TERMINALNO')
    #     user_driver_stat = pd.merge(user_driver_stat, user_direction_stat, how='left', on='TERMINALNO')
    #     user_driver_stat.loc[:, 'Y'] = -1
    #
    #     user_driver_stat.rename(columns={'mean_x': 'mean_speed', 'std_x': 'var_speed', 'max':'max_speed',
    #                                         'mean_y': 'mean_height', 'std_y': 'var_height',
    #                                         'mean': 'mean_direction'}, inplace=True)
    #
    #     return user_driver_stat

    def get_distance(self, data):
        """
        经纬度转距离
        """
        lon_lat_data = data.loc[:, ['TERMINALNO', 'LONGITUDE', 'LATITUDE']]
        user_distance = lon_lat_data.groupby('TERMINALNO', as_index=False)[['LONGITUDE', 'LATITUDE']].first()
        user_distance.loc[:, 'user_distance'] = user_distance.apply(
            lambda row: self.haversine1(row['LONGITUDE'], row['LATITUDE'], 113.9177317, 22.54334333), axis=1)
        user_distance.drop(['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)

        predictors1.append('user_distance')

        res = [user_distance['user_distance'].tolist()]
        return res

    def user_callstate_stat(self, data):
        """
        用户通话状态统计
        """
        callstate_data = data.loc[:, ['TERMINALNO', 'CALLSTATE']]
        user_callstate_stat = callstate_data.groupby(['TERMINALNO'], as_index=False)['CALLSTATE'].count()

        # 用户通话特征
        callstate_data['call_unknow_state'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 0 else 0)
        user_callstate_stat.loc[:, 'call_unknow_state'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'call_unknow_state'].sum()
        user_callstate_stat.loc[:, 'call_unknow_state_rate'] = user_callstate_stat['call_unknow_state'] / \
                                                               user_callstate_stat['CALLSTATE']

        callstate_data['user_call_in_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 2 else 0)
        user_callstate_stat.loc[:, 'user_call_in_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_in_num'].sum()
        user_callstate_stat.loc[:, 'user_call_in_rate'] = user_callstate_stat['user_call_in_num'] / user_callstate_stat[
            'CALLSTATE']

        callstate_data['user_call_out_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 1 else 0)
        user_callstate_stat.loc[:, 'user_call_out_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_out_num'].sum()
        user_callstate_stat.loc[:, 'user_call_out_rate'] = user_callstate_stat['user_call_out_num'] / \
                                                           user_callstate_stat['CALLSTATE']

        callstate_data['user_call_connection_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 3 else 0)
        user_callstate_stat.loc[:, 'user_call_connection_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_connection_num'].sum()
        user_callstate_stat.loc[:, 'user_call_connection_rate'] = user_callstate_stat['user_call_connection_num'] / \
                                                                  user_callstate_stat['CALLSTATE']

        callstate_data['user_call_close_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 4 else 0)
        user_callstate_stat.loc[:, 'user_call_close_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_close_num'].sum()
        user_callstate_stat.loc[:, 'user_call_close_rate'] = user_callstate_stat['user_call_close_num'] / \
                                                             user_callstate_stat['CALLSTATE']

        predictors1.append('call_unknow_state_rate')
        predictors1.append('user_call_in_rate')
        predictors1.append('user_call_out_rate')
        predictors1.append('user_call_connection_rate')
        predictors1.append('user_call_close_rate')
        # predictors1.append('to_dis')

        res = [user_callstate_stat['call_unknow_state_rate'].tolist(),
               user_callstate_stat['user_call_in_rate'].tolist(),
               user_callstate_stat['user_call_out_rate'].tolist(),
               user_callstate_stat['user_call_connection_rate'].tolist(),
               user_callstate_stat['user_call_close_rate'].tolist()
        # lon_first_data['to_dis'].tolist()
        ]

        return res


    def user_direction__stat(self, data):
        """
        用户方向特征统计
        """
        direction_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'DIRECTION']]
        user_direction_stat = direction_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].count()
        direction_data['user_direction_30_num'] = direction_data['DIRECTION'].map(lambda x: 1 if x < 30 else 0)
        user_direction_stat.loc[:, 'user_direction_30_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_30_num'].sum()
        user_direction_stat.loc[:, 'user_direction_30_rate'] = user_direction_stat['user_direction_30_num'] / \
                                                              user_direction_stat['TRIP_ID']
        direction_data['user_direction_30_90_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 30 and x < 90 else 0)
        user_direction_stat.loc[:, 'user_direction_30_90_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_30_90_num'].sum()
        user_direction_stat.loc[:, 'user_direction_30_90_rate'] = user_direction_stat['user_direction_30_90_num'] / \
                                                                 user_direction_stat['TRIP_ID']
        direction_data['user_direction_90_120_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 90 and x < 120 else 0)
        user_direction_stat.loc[:, 'user_direction_90_120_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_90_120_num'].sum()
        user_direction_stat.loc[:, 'user_direction_90_120_rate'] = user_direction_stat['user_direction_90_120_num'] / \
                                                                  user_direction_stat['TRIP_ID']
        direction_data['user_direction_120_180_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 120 and x < 180 else 0)
        user_direction_stat.loc[:, 'user_direction_120_180_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_120_180_num'].sum()
        user_direction_stat.loc[:, 'user_direction_120_180_rate'] = user_direction_stat['user_direction_120_180_num'] / \
                                                                   user_direction_stat['TRIP_ID']
        direction_data['user_direction_180_210_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 180 and x < 210 else 0)
        user_direction_stat.loc[:, 'user_direction_180_210_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction_180_210_num'].sum()
        user_direction_stat.loc[:, 'user_direction_180_210_rate'] = user_direction_stat['user_direction_180_210_num'] / \
                                                                   user_direction_stat[
                                                                       'TRIP_ID']
        direction_data['user_direction__210_360_num'] = direction_data['DIRECTION'].map(
            lambda x: 1 if x > 210 and x < 360 else 0)
        user_direction_stat.loc[:, 'user_direction__210_360_num'] = direction_data.groupby('TERMINALNO', as_index=False)[
            'user_direction__210_360_num'].sum()
        user_direction_stat.loc[:, 'user_direction__210_360_rate'] = user_direction_stat['user_direction__210_360_num'] / \
                                                                    user_direction_stat[
                                                                        'TRIP_ID']

        predictors1.append('user_direction_30_rate')
        predictors1.append('user_direction_30_90_rate')
        predictors1.append('user_direction_90_120_rate')
        predictors1.append('user_direction_120_180_rate')
        predictors1.append('user_direction_180_210_rate')
        predictors1.append('user_direction__210_360_rate')

        res = [ user_direction_stat['user_direction_30_rate'].tolist(),
                user_direction_stat['user_direction_30_90_rate'].tolist(),
                user_direction_stat['user_direction_90_120_rate'].tolist(),
                user_direction_stat['user_direction_120_180_rate'].tolist(),
                user_direction_stat['user_direction_180_210_rate'].tolist(),
                user_direction_stat['user_direction__210_360_rate'].tolist()
                ]
        return res

    def user_height_stat(self, data):
        """
         用户海拔特征统计
        """
        height_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'HEIGHT']]
        user_height_stat = height_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()
        user_height_std = height_data.groupby(['TERMINALNO', 'TRIP_ID'])['HEIGHT'].agg(['std']).reset_index()

        user_height_std['user_height_std'] = user_height_std['std'].replace(np.nan, -100)
        user_height_std.loc[:, 'user_height_nan_num'] = user_height_std['user_height_std'].map(
            lambda x: 1 if x == -100 else 0)
        user_height_stat.loc[:, 'user_height_nan_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_nan_num'].sum()
        user_height_stat.loc[:, 'user_height_nan_rate'] = user_height_stat['user_height_nan_num'] / user_height_stat[
            'TRIP_ID']
        user_height_std.loc[:, 'user_height_2_num'] = user_height_std['std'].map(lambda x: 1 if x < 2 else 0)
        user_height_stat.loc[:, 'user_height_2_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_2_num'].sum()
        user_height_stat.loc[:, 'user_height_std_2_rate'] = user_height_stat['user_height_2_num'] / user_height_stat[
            'TRIP_ID']
        user_height_std.loc[:, 'user_height_2_5_num'] = user_height_std['std'].map(
            lambda x: 1 if x > 2 and x < 5 else 0)
        user_height_stat.loc[:, 'user_height_2_5_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_2_5_num'].sum()
        user_height_stat.loc[:, 'user_height_std_2_5_rate'] = user_height_stat['user_height_2_5_num'] / user_height_stat[
            'TRIP_ID']
        user_height_std.loc[:, 'user_height_5_10_num'] = user_height_std['std'].map(
            lambda x: 1 if x >= 5 and x < 10 else 0)
        user_height_stat.loc[:, 'user_height_5_10_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_5_10_num'].sum()
        user_height_stat.loc[:, 'user_height_std_5_10_rate'] = user_height_stat['user_height_5_10_num'] / \
                                                              user_height_stat[
                                                                  'TRIP_ID']
        user_height_std.loc[:, 'user_height_10_num'] = user_height_std['std'].map(
            lambda x: 1 if x >= 10 else 0)
        user_height_stat.loc[:, 'user_height_10_num'] = user_height_std.groupby(['TERMINALNO'], as_index=False)[
            'user_height_10_num'].sum()
        user_height_stat.loc[:, 'user_height_std_10_rate'] = user_height_stat['user_height_10_num'] / \
                                                            user_height_stat[
                                                                'TRIP_ID']

        predictors1.append('user_height_nan_rate')
        predictors1.append('user_height_std_2_rate')
        predictors1.append('user_height_std_2_5_rate')
        predictors1.append('user_height_std_5_10_rate')
        predictors1.append('user_height_std_10_rate')

        res = [ user_height_stat['user_height_nan_rate'].tolist(),
                user_height_stat['user_height_std_2_rate'].tolist(),
                user_height_stat['user_height_std_2_5_rate'].tolist(),
                user_height_stat['user_height_std_5_10_rate'].tolist(),
                user_height_stat['user_height_std_10_rate'].tolist()
        ]
        return res

    def user_speed_stat(self, data):
        """
        用户速度统计
        """
        speed_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'SPEED']]
        user_speed_stat = speed_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()
        user_speed_std = speed_data.groupby(['TERMINALNO', 'TRIP_ID'])['SPEED'].agg(['std']).reset_index()

        user_speed_std_max = user_speed_std.groupby(['TERMINALNO'], as_index=False)['std'].max()

        user_speed_std['user_speed_std'] = user_speed_std['std'].replace(np.nan, 99999)

        user_speed_std.loc[:, 'user_speed_2_std'] = user_speed_std['user_speed_std'].map(lambda x: 1 if x < 2 else 0)
        user_speed_stat.loc[:, 'user_speed_2_std'] = user_speed_std.groupby(['TERMINALNO'], as_index=False)[
            'user_speed_2_std'].sum()
        user_speed_stat.loc[:, 'user_speed_2_rate'] = user_speed_stat['user_speed_2_std'] / user_speed_stat[
            'TRIP_ID']

        user_speed_std.loc[:, 'user_speed_2_10_std'] = user_speed_std['user_speed_std'].map(
            lambda x: 1 if x > 2 and x < 10 else 0)
        user_speed_stat.loc[:, 'user_speed_2_10_std'] = user_speed_std.groupby(['TERMINALNO'], as_index=False)[
            'user_speed_2_10_std'].sum()
        user_speed_stat.loc[:, 'user_speed_2_10_rate'] = user_speed_stat['user_speed_2_10_std'] / user_speed_stat[
            'TRIP_ID']

        user_speed_std.loc[:, 'user_speed_10_std'] = user_speed_std['user_speed_std'].map(lambda x: 1 if x > 10 else 0)
        user_speed_stat.loc[:, 'user_speed_10_std'] = user_speed_std.groupby(['TERMINALNO'], as_index=False)[
            'user_speed_10_std'].sum()
        user_speed_stat.loc[:, 'user_speed_10_rate'] = user_speed_stat['user_speed_10_std'] / user_speed_stat[
            'TRIP_ID']
        user_speed_stat = pd.merge(user_speed_std_max, user_speed_stat, how='left', on='TERMINALNO')

        user_speed_stat.rename(columns={'std': 'user_rapid_speed_max_std'}, inplace=True)

        predictors1.append('user_rapid_speed_max_std')
        predictors1.append('user_speed_2_rate')
        predictors1.append('user_speed_2_10_rate')
        predictors1.append('user_speed_10_rate')

        res = [user_speed_stat['user_rapid_speed_max_std'].tolist(),
               user_speed_stat['user_speed_2_rate'].tolist(),
               user_speed_stat['user_speed_2_10_rate'].tolist(),
               user_speed_stat['user_speed_10_rate'].tolist(),

        ]
        return res