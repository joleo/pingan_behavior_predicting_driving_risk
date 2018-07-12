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
import pandas as pd
import numpy as np
import datetime
import time
from math import radians, cos, sin, asin, sqrt

class LYFeatureExtraction(object):

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
        user_time_per = user_time_data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].count()

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

        driver_time_colmns = ['hour_5', 'hour_6', 'hour_10', 'hour_15',
                              'hour_7', 'hour_8', 'hour_9', 'hour_17', 'hour_18',
                              'hour_11', 'hour_12', 'hour_13',
                              'hour_14', 'hour_0', 'hour_1', 'hour_2', 'hour_3',
                              'hour_4']

        # 计算每个时间段的比例
        driver_time_num = user_time_data.groupby(['TERMINALNO'], as_index=False)[driver_time_colmns].sum()

        user_time_per.loc[:, 'hour_5_per'] = driver_time_num['hour_5'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_6_per'] = driver_time_num['hour_6'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_10_per'] = driver_time_num['hour_10'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_15_per'] = driver_time_num['hour_15'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_7_per'] = driver_time_num['hour_7'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_8_per'] = driver_time_num['hour_8'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_9_per'] = driver_time_num['hour_9'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_17_per'] = driver_time_num['hour_17'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_18_per'] = driver_time_num['hour_18'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_11_per'] = driver_time_num['hour_11'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_12_per'] = driver_time_num['hour_12'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_13_per'] = driver_time_num['hour_13'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_14_per'] = driver_time_num['hour_14'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_0_per'] = driver_time_num['hour_0'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_1_per'] = driver_time_num['hour_1'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_2_per'] = driver_time_num['hour_2'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_3_per'] = driver_time_num['hour_3'] / user_time_per['TRIP_ID']
        user_time_per.loc[:, 'hour_4_per'] = driver_time_num['hour_4'] / user_time_per['TRIP_ID']

        user_time_per.drop('TRIP_ID', axis=1, inplace=True)

        return user_time_per

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

        return train_data
    def test(self, data):
        test_data = data.loc[:, ['TERMINALNO','DIRECTION','LATITUDE']]
        test = test_data.groupby(['TERMINALNO','LATITUDE'], as_index=False)['DIRECTION'].mean()
        return test

    def user_driver_stat(self, data):
        """
        用户驾驶行为基本特征统计
        """
        # trid 特征
        df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()

        # 驾驶行为特征
        user_speed_stat = data.groupby('TERMINALNO')['SPEED'].agg(
            {"user_speed_mean": "mean", "user_speed_std": "std", "user_speed_max": "max"}).reset_index()
        user_height_stat = data.groupby('TERMINALNO')['HEIGHT'].agg(
            {"user_height_mean": "mean", "user_height_std": "std"}).reset_index()
        user_direction_stat = data.groupby('TERMINALNO')['DIRECTION'].agg(
            {"user_direction_mean": "mean"}).reset_index()
        user_long_stat = data.groupby('TERMINALNO')['LONGITUDE'].agg(
            {"user_lon_std": "std"}).reset_index()
        user_lat_stat = data.groupby('TERMINALNO')['LATITUDE'].agg(
            {"user_lat_std": "std"}).reset_index()
        # 合并
        user_driver_stat = pd.merge(df_user_trid_num, user_speed_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_height_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_direction_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_long_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_lat_stat, how='left', on='TERMINALNO')


        return user_driver_stat

    def user_test_driver_stat(self, data):
        """
        用户驾驶行为基本特征统计
        """
        # trid 特征
        df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()

        # 驾驶行为特征
        # user_speed_stat = data.groupby(['TERMINALNO'])['SPEED'].agg(['mean','std','max']).reset_index()
        # user_height_stat = data.groupby(['TERMINALNO'])['HEIGHT'].agg(['mean','std']).reset_index()
        # user_direction_stat = data.groupby(['TERMINALNO'])['DIRECTION'].agg(['mean']).reset_index()

        user_speed_stat = data.groupby('TERMINALNO')['SPEED'].agg(
            {"user_speed_mean": "mean", "user_speed_std": "std", "user_speed_max": "max"}).reset_index()
        user_height_stat = data.groupby('TERMINALNO')['HEIGHT'].agg(
            {"user_height_mean": "mean", "user_height_std": "std"}).reset_index()
        user_direction_stat = data.groupby('TERMINALNO')['DIRECTION'].agg(
            {"user_direction_mean": "mean"}).reset_index()
        user_long_stat = data.groupby('TERMINALNO')['LONGITUDE'].agg(
            {"user_lon_std": "std"}).reset_index()
        user_lat_stat = data.groupby('TERMINALNO')['LATITUDE'].agg(
            {"user_lat_std": "std"}).reset_index()

        # 合并
        user_driver_stat = pd.merge(df_user_trid_num, user_speed_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_height_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_direction_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_long_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_lat_stat, how='left', on='TERMINALNO')

        user_driver_stat.loc[:, 'Y'] = -1

        # user_driver_stat.rename(columns={'mean_x': 'user_mean_speed', 'std_x': 'user_var_speed', 'max':'user_max_speed',
        #                                     'mean_y': 'user_mean_height', 'std_y': 'user_var_height',
        #                                     'mean': 'user_mean_direction'}, inplace=True)

        return user_driver_stat

    def get_distance(self, data):
        """
        经纬度转距离
        """
        lon_lat_data = data.loc[:, ['TERMINALNO', 'LONGITUDE', 'LATITUDE']]
        user_distance = lon_lat_data.groupby('TERMINALNO', as_index=False)[['LONGITUDE', 'LATITUDE']].first()
        user_distance.loc[:, 'user_distance'] = user_distance.apply(
            lambda row: self.haversine1(row['LONGITUDE'], row['LATITUDE'], 113.9177317, 22.54334333), axis=1)
        user_distance.drop(['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)

        return user_distance



