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

    # def modify_y(self, y):
    #     if y>0 and y<0.1: return y+0.91
    #     elif y<0.2: return y+0.81
    #     elif y<0.3:

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
        user_time_data = data.loc[:, ['TERMINALNO', 'TRIP_ID', 'TIME','HEIGHT','DIRECTION','SPEED']]
        del data
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

        # user_time_data.loc[(user_time_data['hour'] >= 0) & (user_time_data['hour'] <= 6), 'hour_period'] = 1
        # user_time_data.loc[(user_time_data['hour'] >= 18) & (user_time_data['hour'] <= 24), 'hour_period'] = 3

        # user_period_std_height = user_time_data.groupby(['TERMINALNO','hour_period'])['HEIGHT'].agg(
        #     {'user_period_mean_height':'mean'}).reset_index()
        #
        # user_period_1_std_height = user_period_std_height[user_period_std_height['hour_period']==1].rename(columns={'user_period_std_height': 'user_period_1_std_height'})
        # user_period_1_std_height.drop('hour_period', axis=1, inplace=True)
        # user_period_3_std_height = user_period_std_height[user_period_std_height['hour_period']==3].rename(columns={'user_period_std_height': 'user_period_3_std_height'})
        # user_period_3_std_height.drop('hour_period', axis=1, inplace=True)
        #
        # user_period_std_direc = user_time_data.groupby(['TERMINALNO', 'hour_period'])['DIRECTION'].agg(
        #     {'user_period_mean_direc': 'mean'}).reset_index()
        # user_period_1_std_direc = user_period_std_direc[user_period_std_direc['hour_period'] == 1].rename(
        #     columns={'user_period_mean_direc': 'user_period_1_mean_direc'})
        # user_period_1_std_direc.drop('hour_period', axis=1, inplace=True)
        # user_period_3_std_direc = user_period_std_direc[user_period_std_direc['hour_period'] == 3].rename(
        #     columns={'user_period_mean_direc': 'user_period_3_mean_direc'})
        # user_period_3_std_direc.drop('hour_period', axis=1, inplace=True)

        # user_period_std_speed = user_time_data.groupby(['TERMINALNO', 'hour_period'])['SPEED'].agg(
        #     {'user_period_std_speed': 'std'}).reset_index()
        # user_period_1_std_speed = user_period_std_speed[user_period_std_direc['hour_period'] == 1].rename(
        #     columns={'user_period_std_speed': 'user_period_1_std_speed'})
        # user_period_1_std_speed.drop('hour_period', axis=1, inplace=True)
        # user_period_3_std_speed = user_period_std_speed[user_period_std_direc['hour_period'] == 3].rename(
        #     columns={'user_period_std_speed': 'user_period_3_std_speed'})
        # user_period_3_std_speed.drop('hour_period', axis=1, inplace=True)

        # user_time_per = pd.merge(user_time_per, user_period_1_std_height, on="TERMINALNO", how="left")
        # user_time_per = pd.merge(user_time_per, user_period_3_std_height, on="TERMINALNO", how="left")
        # user_time_per = pd.merge(user_time_per, user_period_1_std_direc, on="TERMINALNO", how="left")
        # user_time_per = pd.merge(user_time_per, user_period_3_std_direc, on="TERMINALNO", how="left")
        # user_time_per = pd.merge(user_time_per, user_period_1_std_speed, on="TERMINALNO", how="left")
        # user_time_per = pd.merge(user_time_per, user_period_3_std_speed, on="TERMINALNO", how="left")

        del user_time_data

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

        del driver_time_num
        user_time_per.drop('TRIP_ID', axis=1, inplace=True)



        return user_time_per

    def user_night_stat(self, data):
        """
        夜晚开车和每天用车时长的统计
        """
        train = data.loc[:,['TERMINALNO','TIME','LONGITUDE','HEIGHT']]
        del data
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
        # train['hour_5'] = train.hour.apply(lambda x: 1 if x == 5 else 0, 1)
        # train['hour_6'] = train.hour.apply(lambda x: 1 if x == 6 else 0, 1)
        # train['hour_7'] = train.hour.apply(lambda x: 1 if x == 7 else 0, 1)
        # train['hour_8'] = train.hour.apply(lambda x: 1 if x == 8 else 0, 1)
        # train['hour_9'] = train.hour.apply(lambda x: 1 if x == 9 else 0, 1)
        #
        # train['hour_18'] = train.hour.apply(lambda x: 1 if x == 18 else 0, 1)
        # train['hour_19'] = train.hour.apply(lambda x: 1 if x == 19 else 0, 1)

        train['hour_21'] = train.hour.apply(lambda x: 1 if x == 21 else 0, 1)
        train['hour_22'] = train.hour.apply(lambda x: 1 if x == 22 else 0, 1)
        train['hour_23'] = train.hour.apply(lambda x: 1 if x == 23 else 0, 1)

        train_hour = train.groupby(['TERMINALNO', 'TIME'], as_index=False).count()
        train_hour.TIME = train_hour.TIME.apply(lambda x: str(x)[:10], 1)
        train_day = train_hour.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).count()
        del train_hour
        train_hour_count = train_day.groupby('TERMINALNO')['LONGITUDE'].agg(
            {"hour_count_max": "max",  "hour_count_mean": "mean", "hour_count_std": "std",
             "hour_count_skew": "skew"}).reset_index()

        train_hour_first = train.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).first()
        del train

        train_hour_first.TIME = train_hour_first.TIME.apply(
            lambda x: str(x)[:10], 1)
        train_day_sum = train_hour_first.groupby(
            ['TERMINALNO', 'TIME'], as_index=False).sum()
        train_day_sum['hour_count'] = train_day['LONGITUDE']

        train_day_sum['night_drive_count'] = train_day_sum.apply(lambda x: x['hour_0'] + x['hour_1'] +
                                                                           x['hour_2'] + x['hour_3'] +
                                                                           x['hour_4'] + # x['hour_5'] + x['hour_6'] +
                                                                           # x['hour_7'] + x['hour_8'] + x['hour_9'] +
                                                                           # x['hour_18'] + x['hour_19'] +
                                                                           x['hour_21'] +  x['hour_22'] + x['hour_23']
                                                                            , 1)

        train_day_sum['night_delta'] = train_day_sum['night_drive_count'] / \
                                       train_day_sum['hour_count']

        train_day_sum['night'] = train_day_sum['night_drive_count'].apply(
            lambda x: 1 if x != 0 else 0, 1)
        train_hour_count['night__day_delta'] = train_day_sum.groupby(['TERMINALNO'], as_index=False).sum(
        )['night'] / (train_day_sum.groupby(['TERMINALNO'], as_index=False).count()['HEIGHT'])

        train_night_count = train_day_sum.groupby('TERMINALNO')['night_delta'].agg(
            {"night_count_max": "max",  "night_count_mean": "mean", "night_count_std": "std",
             "night_count_skew": "skew"}).reset_index()
        del train_day_sum
        train_data = pd.merge(
            train_hour_count, train_night_count, on="TERMINALNO", how="left")

        return train_data



    def user_driver_stat(self, data):
        """
        用户驾驶行为基本特征统计
        """
        # trid 特征
        df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()
        # 驾驶行为特征

        user_speed_stat = data.groupby('TERMINALNO')['SPEED'].agg(
            {'user_speed_mean': 'mean','user_speed_std': 'std',
             'user_speed_max': 'max'}).reset_index()
        user_height_stat = data.groupby('TERMINALNO')['HEIGHT'].agg(
            {'user_height_mean': 'mean',  'user_height_std': 'std'}).reset_index()
        user_direction_stat = data.groupby('TERMINALNO')['DIRECTION'].agg(
            {'user_direction_mean': 'mean'}).reset_index()
        # user_height_stat = data.groupby('TERMINALNO')['HEIGHT'].agg(
        #     {'user_height_mean': 'mean', 'user_height_std': 'std'}).reset_index()
        # user_direction_stat = data.groupby('TERMINALNO')['DIRECTION'].agg(
        #     {'user_direction_mean': 'mean'}).reset_index()

        user_long_stat = data.groupby(['TERMINALNO', 'TRIP_ID'])['LONGITUDE'].agg(
            {'user_lon_max': 'max', 'user_lon_min': 'min'}).reset_index()
        user_long_stat['user_long_diff'] = user_long_stat['user_lon_max'] - user_long_stat['user_lon_min']
        user_long_lat_stat = user_long_stat.groupby(['TERMINALNO'], as_index=False)['user_long_diff'].sum()

        user_lat_stat = data.groupby(['TERMINALNO','TRIP_ID'])['LATITUDE'].agg(
            {'user_lat_max': 'max', 'user_lat_min': 'min'}).reset_index()
        user_lat_stat['user_lat_diff'] = user_lat_stat['user_lat_max'] - user_lat_stat['user_lat_min']
        user_long_lat_stat.loc[:, 'user_lat_diff'] = user_lat_stat.groupby(['TERMINALNO'], as_index=False)['user_lat_diff'].sum()


        # 峰度系数是用来反映频数分布曲线顶端尖峭或扁平程度的指标
        user_kurt = data.groupby('TERMINALNO')['HEIGHT','DIRECTION'].apply(lambda  x: x.kurt()).reset_index()
        user_kurt.rename(columns={'HEIGHT': 'user_height_kurt', 'DIRECTION': 'user_direc_kurt'}, inplace=True)
        # 偏态系数是根据众数、中位数与均值各自的性质，通过比较众数或中位数与均值来衡量偏斜度的，即偏态系数是对分布偏斜方向和程度的刻画
        user_skew_1 = data.groupby(['TERMINALNO','TRIP_ID'], as_index=False)['HEIGHT','DIRECTION'].skew()
        # user_quantile = data.groupby('TERMINALNO')['SPEED','HEIGHT','DIRECTION'].\
        #         apply(lambda  x: x.quantile(0.5)).reset_index()
        # user_quantile.rename(
        #     columns={'SPEED': 'user_speed_25_quantile', 'HEIGHT': 'user_height_25_quantile',
        #              'DIRECTION': 'user_direc_25_quantile'}, inplace=True)
        # user_long_stat = data.groupby('TERMINALNO')['LONGITUDE'].agg(
        #     {'user_lon_mean': 'mean', 'user_lon_max': 'max'}).reset_index()
        del data

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
        user_driver_stat = pd.merge(user_driver_stat, user_kurt, how='left', on='TERMINALNO')
        # user_driver_stat = pd.merge(user_driver_stat, user_long_lat_stat, how='left', on='TERMINALNO')

        user_driver_stat = pd.merge(user_driver_stat, user_max_skew, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_min_skew, how='left', on='TERMINALNO')

        return user_driver_stat

    def user_test_driver_stat(self, data):
        """
        用户驾驶行为基本特征统计
        """
        # trid 特征
        df_user_trid_num = data.groupby('TERMINALNO', as_index=False)['TRIP_ID'].max()

        # 驾驶行为特征
        user_speed_stat = data.groupby('TERMINALNO')['SPEED'].agg(
            {'user_speed_mean': 'mean','user_speed_std': 'std', 'user_speed_max': 'max'}).reset_index()
        user_height_stat = data.groupby('TERMINALNO')['HEIGHT'].agg(
            {'user_height_mean': 'mean', 'user_height_std': 'std'}).reset_index()
        user_direction_stat = data.groupby('TERMINALNO')['DIRECTION'].agg(
            {'user_direction_mean': 'mean'}).reset_index()
        # user_height_stat = data.groupby('TERMINALNO')['HEIGHT'].agg(
        #     {'user_height_mean': 'mean','user_height_std': 'std'}).reset_index()
        # user_direction_stat = data.groupby('TERMINALNO')['DIRECTION'].agg(
        #     {'user_direction_mean': 'mean'}).reset_index()
        user_kurt = data.groupby('TERMINALNO')['HEIGHT', 'DIRECTION'].apply(lambda x: x.kurt()).reset_index()
        user_kurt.rename(columns={'HEIGHT': 'user_height_kurt', 'DIRECTION': 'user_direc_kurt'}, inplace=True)
        user_quantile = data.groupby('TERMINALNO')['SPEED', 'HEIGHT', 'DIRECTION'].apply(
            lambda x: x.quantile(0.5)).reset_index()
        user_quantile.rename(
            columns={'SPEED': 'user_speed_25_quantile', 'HEIGHT': 'user_height_25_quantile',
                     'DIRECTION': 'user_direc_25_quantile'}, inplace=True)

        user_kurt.rename(
            columns={'HEIGHT': 'user_height_kurt', 'DIRECTION': 'user_direc_kurt'}, inplace=True)
        user_skew_1 = data.groupby(['TERMINALNO','TRIP_ID'], as_index=False)['HEIGHT','DIRECTION'].skew()
        user_long_stat = data.groupby(['TERMINALNO', 'TRIP_ID'])['LONGITUDE'].agg(
            {'user_lon_max': 'max', 'user_lon_min': 'min'}).reset_index()
        user_long_stat['user_long_diff'] = user_long_stat['user_lon_max'] - user_long_stat['user_lon_min']
        user_long_lat_stat = user_long_stat.groupby(['TERMINALNO'], as_index=False)['user_long_diff'].sum()

        user_lat_stat = data.groupby(['TERMINALNO', 'TRIP_ID'])['LATITUDE'].agg(
            {'user_lat_max': 'max', 'user_lat_min': 'min'}).reset_index()
        user_lat_stat['user_lat_diff'] = user_lat_stat['user_lat_max'] - user_lat_stat['user_lat_min']
        user_long_lat_stat.loc[:, 'user_lat_diff'] = user_lat_stat.groupby(['TERMINALNO'], as_index=False)[
            'user_lat_diff'].sum()

        del data
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
        # user_driver_stat = pd.merge(user_driver_stat, user_long_lat_stat, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_kurt, how='left', on='TERMINALNO')

        user_driver_stat = pd.merge(user_driver_stat, user_max_skew, how='left', on='TERMINALNO')
        user_driver_stat = pd.merge(user_driver_stat, user_min_skew, how='left', on='TERMINALNO')
        user_driver_stat.loc[:, 'Y'] = -1

        return user_driver_stat

    def min_max_normalize(self, df, name):
        """
        归一化
        """
        max_number = df[name].max()
        min_number = df[name].min()
        # assert max_number != min_number, 'max == min in COLUMN {0}'.format(name)
        df[name] = df[name].map(lambda x: float(x - min_number + 1) / float(max_number - min_number + 1))

        return df

    def get_distance(self, data):
        """
        经纬度转距离
        """
        lon_lat_data = data.loc[:, ['TERMINALNO', 'LONGITUDE', 'LATITUDE']]
        del data
        user_distance = lon_lat_data.groupby('TERMINALNO', as_index=False)[['LONGITUDE', 'LATITUDE']].first()
        user_distance.loc[:, 'user_distance'] = user_distance.apply(
            lambda row: self.haversine1(row['LONGITUDE'], row['LATITUDE'], 113.9177317, 22.54334333), axis=1)


        user_distance.drop(['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)
        # user_norm_distance = self.min_max_normalize(user_distance, 'user_distance')

        return user_distance

    def geo_stat(self, data):
        # data = data.loc[:, ['TERMINALNO','TIME','LONGITUDE','LATITUDE']]
        feature = []
        user_id = data['TERMINALNO'].min()
        user_max_lon = data['LONGITUDE'].max()
        user_min_lon = data['LONGITUDE'].min()
        user_max_lat = data['LATITUDE'].max()
        user_min_lat = data['LATITUDE'].min()
        time_duration = (data['TIME'].max() - data['TIME'].min()) / 3600.0 + 1.0
        del data
        user_lon_ratio = (user_max_lon - user_min_lon) / time_duration
        user_lat_ratio = (user_max_lat - user_min_lat) / time_duration

        feature.extend([user_id,  user_lon_ratio, user_lat_ratio])

        return feature

    def user_geo_stat(self, data):
        lat_lon_feature = ['TERMINALNO',
                           'user_lon_ratio', 'user_lat_ratio'
                           ]

        geo_feature = []
        for uid in data['TERMINALNO'].unique():
            hour_fea = data.loc[data['TERMINALNO'] == uid]
            geo_feature.append(self.geo_stat(hour_fea))

        geo_feature = pd.DataFrame(geo_feature, columns=lat_lon_feature)  # , dtype=np.float32)

        return geo_feature