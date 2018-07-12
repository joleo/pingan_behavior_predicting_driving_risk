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
import gc
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

    def is_holiday_week(self, day):
        if day in holiday_data:
            return 1
        else:
            return 0

    def to_datetime(self, value):
        fmt = '%Y-%m-%d'
        value = time.localtime(value)
        dt = time.strftime(fmt, value)
        return dt

    def test(self, data):
        # holiday_data  week_data
        # data['TIME'] = data['TIME'].apply(lambda x: self.timestamp_datetime(x), 1)
        # data['TIME'] = data['TIME'].apply(lambda x: str(x)[:10], 1)
        # data = data.sort_values(by=["TERMINALNO", 'TIME'])
        # data.loc[:, 'isholiday'] = data['TIME'].apply(lambda x: self.isholiday(str(x)))
        res = self.user_holiday_stats(data)
        # relust = self.isholiday(data['TIME'])
        return res
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

        user_time_per.loc[:, 'hour_5_per'] = (driver_time_num['hour_5'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_6_per'] = (driver_time_num['hour_6'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_10_per'] = (driver_time_num['hour_10'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_15_per'] = (driver_time_num['hour_15'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_7_per'] = (driver_time_num['hour_7'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_8_per'] = (driver_time_num['hour_8'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_9_per'] = (driver_time_num['hour_9'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_17_per'] = (driver_time_num['hour_17'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_18_per'] = (driver_time_num['hour_18'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_11_per'] = (driver_time_num['hour_11'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_12_per'] = (driver_time_num['hour_12'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_13_per'] = (driver_time_num['hour_13'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_14_per'] = (driver_time_num['hour_14'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_0_per'] = (driver_time_num['hour_0'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_1_per'] = (driver_time_num['hour_1'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_2_per'] = (driver_time_num['hour_2'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_3_per'] = (driver_time_num['hour_3'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per.loc[:, 'hour_4_per'] = (driver_time_num['hour_4'] / user_time_per['TRIP_ID']).astype(np.float32)
        user_time_per['TERMINALNO'] = user_time_per['TERMINALNO'].astype(np.int32)
        user_time_per.drop('TRIP_ID', axis=1, inplace=True)

        del user_time_data
        gc.collect()
        return user_time_per

    def user_night_stat(self, data):
        """
        夜晚开车和每天用车时长的统计
        """

        # train['TIME'] = train['TIME'].apply(lambda x: self.timestamp_datetime(x), 1)
        data['TIME'] = data['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x), 1)
        data['TIME'] = data['TIME'].apply(lambda x: str(x)[:13], 1)
        train = data.sort_values(by=["TERMINALNO", 'TIME'])
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
            {"hour_count_max": "max",  "hour_count_mean": "mean", "hour_count_std": "std"}).reset_index()

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

        train_night_count = train_day_sum.groupby('TERMINALNO')['night_delta'].agg(
            {"night_count_max": "max",  "night_count_mean": "mean", "night_count_std": "std"}).reset_index()

        feature = pd.merge(
            train_hour_count, train_night_count, on="TERMINALNO", how="left")

        # 假期特征
        data['TIME_2'] = data['TIME'].apply(lambda x: str(x)[:10], 1)
        data['time_flag'] = data['TIME_2'].map(lambda x: self.is_holiday_week(x))
        holiday_count = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].count()
        holiday_num_sum = data.groupby(['TERMINALNO'])['time_flag'].agg(
            {'holiday_num_sum': 'sum'}).reset_index()
        holiday_num_per = pd.merge(holiday_count, holiday_num_sum, how='left', on='TERMINALNO')
        holiday_num_per['holiday_num_per'] = holiday_num_per['holiday_num_sum'] / holiday_num_per['TRIP_ID']
        holiday_num_per.drop(['TRIP_ID', 'holiday_num_sum'], axis=1, inplace=True)

        # holiday_data = data[data['time_flag'] == 1]

        # holiday_speed = holiday_data.groupby(['TERMINALNO'])['SPEED'].agg(
        #     {'holiday_mean_speed': 'mean', 'holiday_speed_std': 'std'}).reset_index()
        # holiday_height = holiday_data.groupby(['TERMINALNO'])['HEIGHT'].agg(
        #     {'holiday_mean_height': 'mean', 'holiday_height_std': 'std'}).reset_index()
        # holiday_direc = holiday_data.groupby(['TERMINALNO'])['DIRECTION'].agg(
        #     {'holiday_mean_direc': 'mean', 'holiday_direc_std': 'std'}).reset_index()
        #
        # holiday_feature = pd.merge(holiday_speed, holiday_height, how='left', on='TERMINALNO')
        # holiday_feature = pd.merge(holiday_feature, holiday_direc, how='left', on='TERMINALNO')
        # holiday_feature = pd.merge(holiday_feature, train_data, how='left', on='TERMINALNO')
        # feature = pd.merge(train_data, holiday_num_per, how='left', on='TERMINALNO')

        feature['TERMINALNO'] = feature['TERMINALNO'].astype(np.int32)
        feature['hour_count_max'] = feature['hour_count_max'].astype(np.int32)
        feature['hour_count_mean'] = feature['hour_count_mean'].astype(np.int32)
        feature['hour_count_std'] = feature['hour_count_std'].astype(np.float32)
        feature['night_count_max'] = feature['night_count_max'].astype(np.float32)
        feature['night_count_mean'] = feature['night_count_mean'].astype(np.float32)
        feature['night_count_std'] = feature['night_count_std'].astype(np.float32)
        # feature['holiday_num_per'] = feature['holiday_num_per'].astype(np.float32)

        del train_hour_count
        del train_night_count
        del holiday_count
        del holiday_num_sum
        gc.collect()
        return feature

    def user_driver_stat(self, data):
        """
        用户驾驶行为基本特征统计
        """
        # trid 特征
        df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()
        user_time_per = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].count()

        # 驾驶行为特征

        user_speed_stat = data.groupby('TERMINALNO')['SPEED'].agg(
            {'user_speed_mean': 'mean', 'user_speed_std': 'std', 'user_speed_max': 'max'}).reset_index()
        user_height_stat = data.groupby('TERMINALNO')['HEIGHT'].agg(
            {'user_height_mean': 'mean', 'user_height_std': 'std'}).reset_index()
        user_direction_stat = data.groupby('TERMINALNO')['DIRECTION'].agg(
            {'user_direction_mean': 'mean'}).reset_index()
        user_long_stat = data.groupby('TERMINALNO')['LONGITUDE'].agg(
            {'user_lon_mean': 'mean', 'user_lon_std': 'std'}).reset_index()

        # 峰度系数是用来反映频数分布曲线顶端尖峭或扁平程度的指标
        user_kurt = data.groupby('TERMINALNO')['HEIGHT','DIRECTION'].apply(lambda  x: x.kurt()).reset_index()
        # user_kurt = data.groupby(['TERMINALNO', 'TRIP_ID'], as_index=False)['HEIGHT','DIRECTION'].apply(lambda  x: x.kurt())#.max().reset_index()
        # user_kurt_1 = user_kurt.groupby(['TERMINALNO'], as_index=False)['HEIGHT','DIRECTION'].max().reset_index()
        user_kurt.rename(columns={'HEIGHT': 'user_height_kurt', 'DIRECTION': 'user_direc_kurt'}, inplace=True)
        # 偏态系数是根据众数、中位数与均值各自的性质，通过比较众数或中位数与均值来衡量偏斜度的，即偏态系数是对分布偏斜方向和程度的刻画
        user_skew_1 = data.groupby(['TERMINALNO','TRIP_ID'], as_index=False)['HEIGHT','DIRECTION'].skew()
        user_max_skew = user_skew_1.groupby('TERMINALNO')['HEIGHT','DIRECTION'].max().reset_index()
        user_min_skew = user_skew_1.groupby(['TERMINALNO'])['HEIGHT', 'DIRECTION'].min().reset_index()
        user_max_skew.rename(
            columns={'HEIGHT': 'user_height_max_skew', 'DIRECTION': 'user_direc_max_skew'}, inplace=True)
        user_min_skew.rename(
            columns={'HEIGHT': 'user_height_min_skew', 'DIRECTION': 'user_direc_min_skew'}, inplace=True)

        # 合并
        user_driver_stat = pd.merge(df_user_trid_num, user_speed_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_height_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_direction_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_long_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_kurt, how='left', on='TERMINALNO')

        user_driver_stat = pd.merge(user_driver_stat, user_max_skew, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_min_skew, how='left', on='TERMINALNO')

        del user_long_stat
        del df_user_trid_num
        del user_speed_stat
        del user_height_stat
        del user_direction_stat
        del user_kurt
        del user_max_skew
        del user_min_skew


        # 类型转换
        user_driver_stat['TERMINALNO'] = user_driver_stat['TERMINALNO'].astype(np.int32)
        user_driver_stat['TRIP_ID'] = user_driver_stat['TRIP_ID'].astype(np.int16)
        user_driver_stat['user_speed_mean'] = user_driver_stat['user_speed_mean'].astype(np.float32)
        user_driver_stat['user_speed_std'] = user_driver_stat['user_speed_std'].astype(np.float32)
        user_driver_stat['user_speed_max'] = user_driver_stat['user_speed_max'].astype(np.float32)
        user_driver_stat['user_height_mean'] = user_driver_stat['user_height_mean'].astype(np.float32)
        user_driver_stat['user_height_std'] = user_driver_stat['user_height_std'].astype(np.float32)
        user_driver_stat['user_direction_mean'] = user_driver_stat['user_direction_mean'].astype(np.float32)
        user_driver_stat['user_lon_mean'] = user_driver_stat['user_lon_mean'].astype(np.float32)
        user_driver_stat['user_lon_std'] = user_driver_stat['user_lon_std'].astype(np.float32)
        user_driver_stat['user_height_kurt'] = user_driver_stat['user_height_kurt'].astype(np.float32)
        user_driver_stat['user_direc_kurt'] = user_driver_stat['user_direc_kurt'].astype(np.float32)
        user_driver_stat['user_height_max_skew'] = user_driver_stat['user_height_max_skew'].astype(np.float32)
        user_driver_stat['user_direc_max_skew'] = user_driver_stat['user_direc_max_skew'].astype(np.float32)
        user_driver_stat['user_height_min_skew'] = user_driver_stat['user_height_min_skew'].astype(np.float32)
        user_driver_stat['user_direc_min_skew'] = user_driver_stat['user_direc_min_skew'].astype(np.float32)

        gc.collect()
        return user_driver_stat

    def user_test_driver_stat(self, data):
        """
        用户驾驶行为基本特征统计
        """
        # trid 特征
        df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()

        # 驾驶行为特征
        user_speed_stat = data.groupby('TERMINALNO')['SPEED'].agg(
            {'user_speed_mean': 'mean', 'user_speed_std': 'std', 'user_speed_max': 'max'}).reset_index()
        user_height_stat = data.groupby('TERMINALNO')['HEIGHT'].agg(
            {'user_height_mean': 'mean', 'user_height_std': 'std'}).reset_index()
        user_direction_stat = data.groupby('TERMINALNO')['DIRECTION'].agg(
            {'user_direction_mean': 'mean'}).reset_index()
        user_long_stat = data.groupby('TERMINALNO')['LONGITUDE'].agg(
            {'user_lon_mean': 'mean', 'user_lon_std': 'std'}).reset_index()
        user_kurt = data.groupby('TERMINALNO')['HEIGHT', 'DIRECTION'].apply(lambda x: x.kurt()).reset_index()
        user_kurt.rename(columns={'HEIGHT': 'user_height_kurt', 'DIRECTION': 'user_direc_kurt'}, inplace=True)

        # user_kurt = data.groupby(['TERMINALNO', 'TRIP_ID'], as_index=False)['HEIGHT','DIRECTION'].apply(lambda  x: x.kurt())#.max().reset_index()
        # user_kurt_1 = user_kurt.groupby(['TERMINALNO'], as_index=False)['HEIGHT', 'DIRECTION'].max().reset_index()
        user_kurt.rename(
            columns={'HEIGHT': 'user_height_kurt', 'DIRECTION': 'user_direc_kurt'}, inplace=True)
        user_skew_1 = data.groupby(['TERMINALNO','TRIP_ID'], as_index=False)['HEIGHT','DIRECTION'].skew()
        user_max_skew = user_skew_1.groupby('TERMINALNO')['HEIGHT', 'DIRECTION'].max().reset_index()
        user_min_skew = user_skew_1.groupby(['TERMINALNO'])['HEIGHT', 'DIRECTION'].min().reset_index()
        user_max_skew.rename(
            columns={'HEIGHT': 'user_height_max_skew', 'DIRECTION': 'user_direc_max_skew'}, inplace=True)
        user_min_skew.rename(
            columns={'HEIGHT': 'user_height_min_skew', 'DIRECTION': 'user_direc_min_skew'}, inplace=True)

        # 合并
        user_driver_stat = pd.merge(df_user_trid_num, user_speed_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_height_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_direction_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_long_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_kurt, how='left', on='TERMINALNO')

        user_driver_stat = pd.merge(user_driver_stat, user_max_skew, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_min_skew, how='left', on='TERMINALNO')

        user_driver_stat.loc[:, 'Y'] = -1
        del user_long_stat
        del df_user_trid_num
        del user_speed_stat
        del user_height_stat
        del user_direction_stat
        del user_kurt
        del user_max_skew
        del user_min_skew

        # 类型转换
        user_driver_stat['TERMINALNO'] = user_driver_stat['TERMINALNO'].astype(np.int32)
        user_driver_stat['TRIP_ID'] = user_driver_stat['TRIP_ID'].astype(np.int16)
        user_driver_stat['user_speed_mean'] = user_driver_stat['user_speed_mean'].astype(np.float32)
        user_driver_stat['user_speed_std'] = user_driver_stat['user_speed_std'].astype(np.float32)
        user_driver_stat['user_speed_max'] = user_driver_stat['user_speed_max'].astype(np.float32)
        user_driver_stat['user_height_mean'] = user_driver_stat['user_height_mean'].astype(np.float32)
        user_driver_stat['user_height_std'] = user_driver_stat['user_height_std'].astype(np.float32)
        user_driver_stat['user_direction_mean'] = user_driver_stat['user_direction_mean'].astype(np.float32)
        user_driver_stat['user_lon_mean'] = user_driver_stat['user_lon_mean'].astype(np.float32)
        user_driver_stat['user_lon_std'] = user_driver_stat['user_lon_std'].astype(np.float32)
        user_driver_stat['user_height_kurt'] = user_driver_stat['user_height_kurt'].astype(np.float32)
        user_driver_stat['user_direc_kurt'] = user_driver_stat['user_direc_kurt'].astype(np.float32)
        user_driver_stat['user_height_max_skew'] = user_driver_stat['user_height_max_skew'].astype(np.float32)
        user_driver_stat['user_direc_max_skew'] = user_driver_stat['user_direc_max_skew'].astype(np.float32)
        user_driver_stat['user_height_min_skew'] = user_driver_stat['user_height_min_skew'].astype(np.float32)
        user_driver_stat['user_direc_min_skew'] = user_driver_stat['user_direc_min_skew'].astype(np.float32)

        gc.collect()
        return user_driver_stat

    # def user_speed_kurt(self, data):
    #     # 速度的峰度系数
    #     speed_kurt = data.groupby(['TERMINALNO', 'TRIP_ID'])['SPEED'].apply(lambda x: x.kurt()).reset_index()
    #     speed_kurt.rename(columns={'SPEED': 'speed_kurt'}, inplace=True)
    #     # user_speed_max_kurt = speed_kurt.groupby(['TERMINALNO'])['speed_kurt'].agg(
    #     #     {'user_speed_max_kurt': 'max'}).reset_index()
    #     user_speed_min_kurt = speed_kurt.groupby(['TERMINALNO'])['speed_kurt'].agg(
    #         {'user_speed_min_kurt': 'min'}).reset_index()
    #     user_speed_mean_kurt = speed_kurt.groupby(['TERMINALNO'])['speed_kurt'].agg(
    #         {'user_speed_mean_kurt': 'mean'}).reset_index()
    #
    #     user_speed_kurt = pd.merge(user_speed_mean_kurt, user_speed_min_kurt, how='left', on='TERMINALNO')
    #     # user_speed_kurt = pd.merge(user_speed_min_kurt, user_speed_mean_kurt, how='left', on='TERMINALNO')
    #
    #     return user_speed_kurt

    def get_distance(self, data):
        """
        经纬度转距离
        """
        lon_lat_data = data.loc[:, ['TERMINALNO', 'LONGITUDE', 'LATITUDE']]
        user_distance = lon_lat_data.groupby('TERMINALNO', as_index=False)[['LONGITUDE', 'LATITUDE']].first()
        user_distance.loc[:, 'user_distance'] = user_distance.apply(
            lambda row: self.haversine1(row['LONGITUDE'], row['LATITUDE'], 113.9177317, 22.54334333), axis=1)
        user_distance.drop(['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)

        user_distance['TERMINALNO'] = user_distance['TERMINALNO'].astype(np.int32)
        user_distance['user_distance'] = user_distance['user_distance'].astype(np.float32)

        del lon_lat_data
        gc.collect()
        return user_distance



    # def get_similar_position(self, data):
    #     from sklearn.cluster import KMeans
    #
    #     similar_position_labels = self.get_distance(data)
    #     Kmeans_mod = KMeans(n_clusters=100)
    #     similar_position_labels['similar_position_labels'] = Kmeans_mod.fit(
    #         similar_position_labels['user_distance'].reshape(-1, 1)).labels_
    #     similar_position_labels.drop(['user_distance'], axis=1, inplace=True)
    #
    #     return similar_position_labels

    def user_callstate_stat(self, data):
        """
        用户通话状态统计
        """
        callstate_data = data.loc[:, ['TERMINALNO', 'CALLSTATE']]
        user_callstate_stat = callstate_data.groupby(['TERMINALNO'], as_index=False)['CALLSTATE'].count()

        # 未知状态
        callstate_data['call_unknow_state'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x == 0 else 0)

        user_callstate_stat.loc[:, 'call_unknow_state'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'call_unknow_state'].sum()
        user_callstate_stat.loc[:, 'call_unknow_state_per'] = user_callstate_stat['call_unknow_state'] / \
                                                               user_callstate_stat['CALLSTATE']

        # 打电话状态
        callstate_data['user_call_num'] = callstate_data['CALLSTATE'].map(lambda x: 1 if x != 0  else 0)
        user_callstate_stat.loc[:, 'user_call_num'] = callstate_data.groupby(['TERMINALNO'], as_index=False)[
            'user_call_num'].sum()
        user_callstate_stat.loc[:, 'user_call_num_per'] = user_callstate_stat['user_call_num'] / \
                                                             user_callstate_stat['CALLSTATE']
        user_callstate_stat.drop(['CALLSTATE', 'call_unknow_state', 'user_call_num'], axis=1, inplace=True)

        return user_callstate_stat


holiday_data = ['2016-09-15','2016-09-16','2016-09-17','2016-10-01','2016-10-02','2016-10-03'
                 , '2016-10-04','2016-10-05','2016-10-06','2016-10-07','2016-12-24','2016-12-25'
                 , '2016-12-03','2016-12-31','2017-01-01','2016-07-02','2016-07-03','2016-07-09','2016-07-10','2016-07-16','2016-07-17',
                 '2016-07-23''2016-07-24','2016-07-30','2016-07-31',
                 '2016-08-06','2016-08-07''2016-08-13''2016-08-14','2016-08-20','2016-08-21',
                 '2016-08-27','2016-08-28',
                 '2016-09-03','2016-09-04','2016-09-10','2016-09-11','2016-09-04','2016-09-24'
                 , '2016-09-25','2016-10-15','2016-10-16','2016-10-22','2016-10-23','2016-10-29'
                 , '2016-10-30','2016-11-05','2016-11-06','2016-11-12','2016-11-13','2016-11-19'
                  , '2016-11-20','2016-11-26','2016-11-27','2016-12-03','2016-12-04','2016-12-10'
                  , '2016-12-11','2016-12-17','2016-12-18' ]



