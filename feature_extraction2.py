# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_extraction2
   Description :
   Author :       Administrator
   date：          2018/5/20 0020
-------------------------------------------------
   Change Activity:
                   2018/5/20 0020:
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

    def time_num(self, driver_time_data):
        # 行程量最多的时段
        driver_time_num = driver_time_data.value_counts()
        max_num = driver_time_num.index[0]
        min_num = driver_time_num.index[-1]
        # 各时段录量均值、方差、最大、最小值
        driver_time_mean_num = driver_time_num.mean()
        driver_time_std_num = driver_time_num.std()
        driver_time_max_num = driver_time_num.max()
        driver_time_min_num = driver_time_num.min()

        return max_num, min_num, driver_time_mean_num, driver_time_std_num, \
               driver_time_max_num, driver_time_min_num

    def hour_fea(self, data):
        """
        时间特征
        """
        data = data.loc[data['SPEED'] >= 0]
        data = data.drop_duplicates()
        feature = []
        record_num = data.shape[0]
        feature.append(record_num)

        data['hour'] = data['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)
        data['day'] = data['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).day)
        data['month'] = data['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).month)
        data['weekday'] = data['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).weekday())

        data['hour_period'] = 0
        data.loc[(data['hour'] >= 0) & (data['hour'] <= 6), 'hour_period'] = 1
        data.loc[(data['hour'] >= 10) & (data['hour'] <= 16), 'hour_period'] = 2
        data.loc[(data['hour'] >= 21) & (data['hour'] <= 24), 'hour_period'] = 3

        feature.extend(self.time_num(data['hour']))
        feature.extend(self.time_num(data['day']))
        feature.extend(self.time_num(data['weekday']))
        feature.extend(self.time_num(data['month']))
        feature.extend(self.time_num(data['hour_period']))

        # 行程数量(以hour为单位)
        hour_counts = data['hour'].value_counts()
        hour_num = np.zeros(24, dtype=np.float32)
        hour_num[hour_counts.index] = hour_counts
        feature.extend(hour_num)
        del hour_counts, hour_num


        # print('>>>>>>>>>>>>>>>>>>>>>>>>>时间特征<<<<<<<<<<<<<<<<<<<<<<<<<<')
        hour_speed_group = data['SPEED'].groupby(data['hour'])
        # 平均速度(以hour为单位)
        hour_mean_speed = hour_speed_group.mean()
        driver_hour_mean_speed = np.zeros(24, dtype=np.float32)
        driver_hour_mean_speed[hour_mean_speed.index] = hour_mean_speed
        feature.extend(driver_hour_mean_speed)
        del hour_mean_speed

        # 速度方差(以hour为单位)
        hour_std_speed = hour_speed_group.std()
        driver_hour_std_speed = np.zeros(24, dtype=np.float32)
        driver_hour_std_speed[hour_std_speed.index] = hour_std_speed
        feature.extend(driver_hour_std_speed)
        del hour_std_speed

        hour_height_group = data['HEIGHT'].groupby(data['hour'])
        # 海拔均值(以hour为单位)
        hour_mean_height = hour_height_group.mean()
        driver_hour_mean_height = np.zeros(24, dtype=np.float32)
        driver_hour_mean_height[hour_mean_height.index] = hour_mean_height
        feature.extend(driver_hour_mean_height)
        del hour_mean_height

        # 海拔方差(以hour为单位)
        hour_std_height = hour_height_group.std()
        driver_hour_std_height = np.zeros(24, dtype=np.float32)
        driver_hour_std_height[hour_std_height.index] = hour_std_height
        feature.extend(driver_hour_std_height)
        del hour_std_height

        # 海拔偏度(以hour为单位)
        hour_skew_height = hour_height_group.skew()
        driver_hour_skew_height = np.zeros(24, dtype=np.float32)
        driver_hour_skew_height[hour_skew_height.index] = hour_skew_height
        feature.extend(driver_hour_skew_height)
        del hour_skew_height

        # 海拔峰度(以hour为单位)
        hour_kurt_height = hour_height_group.apply(lambda x: x.kurt())
        driver_hour_kurt_height = np.zeros(24, dtype=np.float32)
        driver_hour_kurt_height[hour_kurt_height.index] = hour_kurt_height
        feature.extend(driver_hour_kurt_height)
        del hour_kurt_height

        hour_direc_group = data['DIRECTION'].groupby(data['hour'])
        # 方向均值(以hour为单位)
        hour_mean_direc = hour_direc_group.mean()
        driver_hour_mean_direc = np.zeros(24, dtype=np.float32)
        driver_hour_mean_direc[hour_mean_direc.index] = hour_mean_direc
        feature.extend(driver_hour_mean_direc)
        del hour_mean_direc

        # 方向偏度(以hour为单位)
        hour_skew_direc = hour_direc_group.skew()
        driver_hour_skew_direc = np.zeros(24, dtype=np.float32)
        driver_hour_skew_direc[hour_skew_direc.index] = hour_skew_direc
        feature.extend(driver_hour_skew_direc)
        del hour_skew_direc

        # 方向峰度(以hour为单位)
        hour_kurt_direc = hour_direc_group.apply(lambda x: x.kurt())
        driver_hour_kurt_direc = np.zeros(24, dtype=np.float32)
        driver_hour_kurt_direc[hour_kurt_direc.index] = hour_kurt_direc
        feature.extend(driver_hour_kurt_direc)
        del hour_kurt_direc

        hour_period_speed_group = data['SPEED'].groupby(data['hour_period'])
        # 速度均值（以hour_period为单位）
        hour_period_mean_speed= hour_period_speed_group.mean()
        driver_period_mean_speed = np.zeros(4, dtype=np.float32)
        driver_period_mean_speed[hour_period_mean_speed.index] = hour_period_mean_speed
        feature.extend(driver_period_mean_speed)
        del hour_period_mean_speed

        # 速度方差（以hour_period为单位）
        hour_period_std_speed = hour_period_speed_group.std()
        driver_period_std_speed = np.zeros(4, dtype=np.float32)
        driver_period_std_speed[hour_period_std_speed.index] = hour_period_std_speed
        feature.extend(driver_period_std_speed)
        del hour_period_std_speed

        hour_period_height_group = data['HEIGHT'].groupby(data['hour_period'])
        # 海拔均值（以hour_period为单位）
        hour_period_mean_height = hour_period_height_group.mean()
        driver_period_mean_height = np.zeros(4, dtype=np.float32)
        driver_period_mean_height[hour_period_mean_height.index] = hour_period_mean_height
        feature.extend(driver_period_mean_height)
        del hour_period_mean_height

        # 海拔方差（以hour_period为单位）
        hour_period_std_height = hour_period_height_group.std()
        driver_period_std_height = np.zeros(4, dtype=np.float32)
        driver_period_std_height[hour_period_std_height.index] = hour_period_std_height
        feature.extend(driver_period_std_height)
        del hour_period_std_height

        hour_period_direc_group = data['DIRECTION'].groupby(data['hour_period'])
        # 方向均值（以hour_period为单位）
        hour_period_mean_direc = hour_period_direc_group.mean()
        driver_period_mean_direc = np.zeros(4, dtype=np.float32)
        driver_period_mean_direc[hour_period_mean_direc.index] = hour_period_mean_direc
        feature.extend(driver_period_mean_direc)
        del hour_period_mean_direc

        # 方向偏度（以hour_period为单位）
        hour_period_skew_direc = hour_period_direc_group.skew()
        driver_period_skew_direc = np.zeros(4, dtype=np.float32)
        driver_period_skew_direc[hour_period_skew_direc.index] = hour_period_skew_direc
        feature.extend(driver_period_skew_direc)
        del hour_period_skew_direc

        # 方向峰度 （以hour_period为单位）
        hour_period_kurt_direc = hour_period_direc_group.apply(lambda x: x.kurt())
        driver_period_kurt_direc = np.zeros(4, dtype=np.float32)
        driver_period_kurt_direc[hour_period_kurt_direc.index] = hour_period_kurt_direc
        feature.extend(driver_period_kurt_direc)
        del hour_period_kurt_direc


        # print('>>>>>>>>>>>>>>>>>>>>>>>>>状态特征<<<<<<<<<<<<<<<<<<<<<<<<<<')

        # callstate_speed_group = data['SPEED'].groupby(data['CALLSTATE'])
        # 各状态速度均值
        # callstate_mean_speed = callstate_speed_group.mean()
        # driver_callstate_speed = np.zeros(5, dtype=np.float32)
        # driver_callstate_speed[callstate_mean_speed.index] = callstate_mean_speed
        # feature.extend(driver_callstate_speed)
        # del callstate_mean_speed

        # 各状态速度标准差
        # callstate_std_speed = callstate_speed_group.std()
        # driver_callstate_speed = np.zeros(5, dtype=np.float32)
        # driver_callstate_speed[callstate_std_speed.index] = callstate_std_speed
        # feature.extend(driver_callstate_speed)
        # del callstate_std_speed

        # [5] 状态特征
        # calllstate_per = data['CALLSTATE'].value_counts() / record_num
        # driver_calllstate_per = np.zeros(5, dtype=np.float32)
        # driver_calllstate_per[calllstate_per.index] = calllstate_per
        # feature.extend(driver_calllstate_per)
        # del calllstate_per


        '''
        # 基本特征
        '''
        # 速度特征
        user_mean_speed = data['SPEED'].mean()
        user_max_speed = data['SPEED'].max()
        user_std_speed = data['SPEED'].std()
        user_skew_speed = data['SPEED'].skew()

        # 高度特征
        # user_mean_height = data['HEIGHT'].mean()
        # user_std_height = data['HEIGHT'].std()
        # user_skew_height = data['HEIGHT'].skew()
        # user_kurt_height = data['HEIGHT'].kurt()

        # 方向特征
        # user_mean_direc = data['DIRECTION'].mean()
        # user_skew_direc= data['DIRECTION'].skew()
        # user_kurt_direc = data['DIRECTION'].kurt()

        # feature.extend([user_mean_speed, user_max_speed, user_std_speed,user_mean_height,user_std_height,
        #                 user_skew_height,user_kurt_height,user_mean_direc,user_skew_direc,user_kurt_direc])
        feature.extend([user_mean_speed, user_max_speed, user_std_speed,user_skew_speed])

        # 经纬度特征
        user_max_lon = data['LONGITUDE'].max()
        user_min_lon = data['LONGITUDE'].min()
        user_max_lat = data['LATITUDE'].max()
        user_min_lat = data['LATITUDE'].min()
        time_duration = (data['TIME'].max() - data['TIME'].min()) / 3600.0 + 1.0
        user_lon_ratio = (user_max_lon - user_min_lon) / time_duration
        user_lat_ratio = (user_max_lat - user_min_lat) / time_duration
        data_sort = data.sort_values(by='TIME')
        start_lon = data_sort.iloc[0]['LONGITUDE']
        start_lat = data_sort.iloc[0]['LATITUDE']
        user_distance = self.haversine1(start_lon, start_lat, 113.9177317, 22.54334333)  # 距离某一点的距离

        # 将经纬度取整，拼接起来作为地块编码
        data_sort['geo_code'] = data_sort[['LONGITUDE', 'LATITUDE']].apply(lambda p: int(p[0]) * 100 + int(p[1]), axis=1)
        geo_code_num = data_sort['geo_code'].value_counts()
        user_max_geo_code = geo_code_num.index[0]
        geo_code_per = geo_code_num / data_sort.shape[0]
        geo_code_max_per = geo_code_per.iloc[0]
        user_loc_entropy = ((-1) * geo_code_per * np.log2(geo_code_per)).sum()
        user_loc_num = len(geo_code_per)

        feature.extend([user_max_lon, user_min_lon, user_max_lat, user_min_lat, user_lon_ratio, user_lat_ratio,
                        user_distance, user_max_geo_code, geo_code_max_per, user_loc_entropy,user_loc_num])

        return feature

    def user_hour_feature(self, data):

        time_feature_col = ['record_num',
                        'max_hour_num','min_hour_mun','driver_hour_mean_num','driver_hour_std_num','driver_hour_max_num','driver_hour_min_num',
                        'max_day_num','min_day_mun','driver_day_mean_num','driver_day_std_num','driver_day_max_num','driver_day_min_num',
                        'max_weekday_num','min_weekday_mun','driver_weekday_mean_num','driver_weekday_std_num','driver_weekday_max_num','driver_weekday_min_num',
                        'max_month_num','min_month_mun','driver_month_mean_num','driver_month_std_num','driver_month_max_num','driver_month_min_num',
                        'max_hour_period_num','min_hour_period_mun','driver_hour_period_mean_num','driver_hour_period_std_num','driver_hour_period_max_num','driver_hour_period_min_num',

                        'hour_0_trip', 'hour_1_trip','hour_2_trip','hour_3_trip','hour_4_trip','hour_5_trip','hour_6_trip','hour_7_trip','hour_8_trip'
                        ,'hour_9_trip','hour_10_trip','hour_11_trip','hour_12_trip','hour_13_trip','hour_14_trip','hour_15_trip','hour_16_trip','hour_17_trip','hour_18_trip','hour_19_trip','hour_20_trip','hour_21_trip','hour_22_trip','hour_23_trip',

                        'hour_0_mean_speed', 'hour_1_mean_speed','hour_2_mean_speed','hour_3_mean_speed','hour_4_mean_speed','hour_5_mean_speed','hour_6_mean_speed','hour_7_mean_speed','hour_8_mean_speed','hour_9_mean_speed','hour_10_mean_speed','hour_11_mean_speed','hour_12_mean_speed','hour_13_mean_speed','hour_14_mean_speed','hour_15_mean_speed','hour_16_mean_speed','hour_17_mean_speed','hour_18_mean_speed','hour_19_mean_speed','hour_20_mean_speed','hour_21_mean_speed','hour_22_mean_speed','hour_23_mean_speed',

                        'hour_0_std_speed', 'hour_1_std_speed','hour_2_std_speed','hour_3_std_speed','hour_4_std_speed','hour_5_std_speed','hour_6_std_speed','hour_7_std_speed','hour_8_std_speed','hour_9_std_speed','hour_10_std_speed','hour_11_std_speed','hour_12_std_speed','hour_13_std_speed','hour_14_std_speed','hour_15_std_speed','hour_16_std_speed','hour_17_std_speed','hour_18_std_speed','hour_19_std_speed','hour_20_std_speed','hour_21_std_speed','hour_22_std_speed','hour_23_std_speed',

                        'hour_0_mean_height', 'hour_1_mean_height','hour_2_mean_height','hour_3_mean_height','hour_4_mean_height','hour_5_mean_height','hour_6_mean_height','hour_7_mean_height','hour_8_mean_height','hour_9_mean_height','hour_10_mean_height','hour_11_mean_height','hour_12_mean_height','hour_13_mean_height','hour_14_mean_height','hour_15_mean_height','hour_16_mean_height','hour_17_mean_height','hour_18_mean_height','hour_19_mean_height','hour_20_mean_height','hour_21_mean_height','hour_22_mean_height','hour_23_mean_height',
                        'hour_0_std_height', 'hour_1_std_height','hour_2_std_height','hour_3_std_height','hour_4_std_height','hour_5_std_height','hour_6_std_height','hour_7_std_height','hour_8_std_height','hour_9_std_height','hour_10_std_height','hour_11_std_height','hour_12_std_height','hour_13_std_height','hour_14_std_height','hour_15_std_height','hour_16_std_height','hour_17_std_height','hour_18_std_height','hour_19_std_height','hour_20_std_height','hour_21_std_height','hour_22_std_height','hour_23_std_height',
                        'hour_0_skew_height', 'hour_1_skew_height','hour_2_skew_height','hour_3_skew_height','hour_4_skew_height','hour_5_skew_height','hour_6_skew_height','hour_7_skew_height','hour_8_skew_height','hour_9_skew_height','hour_10_skew_height','hour_11_skew_height','hour_12_skew_height','hour_13_skew_height','hour_14_skew_height','hour_15_skew_height','hour_16_skew_height','hour_17_skew_height','hour_18_skew_height','hour_19_skew_height','hour_20_skew_height','hour_21_skew_height','hour_22_skew_height','hour_23_skew_height',
                        'hour_0_kurt_height', 'hour_1_kurt_height','hour_2_kurt_height','hour_3_kurt_height','hour_4_kurt_height','hour_5_kurt_height','hour_6_kurt_height','hour_7_kurt_height','hour_8_kurt_height','hour_9_kurt_height','hour_10_kurt_height','hour_11_kurt_height','hour_12_kurt_height','hour_13_kurt_height','hour_14_kurt_height','hour_15_kurt_height','hour_16_kurt_height','hour_17_kurt_height','hour_18_kurt_height','hour_19_kurt_height','hour_20_kurt_height','hour_21_kurt_height','hour_22_kurt_height','hour_23_kurt_height',

                        'hour_0_mean_direction', 'hour_1_mean_direction','hour_2_mean_direction','hour_3_mean_direction','hour_4_mean_direction','hour_5_mean_direction','hour_6_mean_direction','hour_7_mean_direction','hour_8_mean_direction','hour_9_mean_direction','hour_10_mean_direction','hour_11_mean_direction','hour_12_mean_direction','hour_13_mean_direction','hour_14_mean_direction','hour_15_mean_direction','hour_16_mean_direction','hour_17_mean_direction','hour_18_mean_direction','hour_19_mean_direction','hour_20_mean_direction','hour_21_mean_direction','hour_22_mean_direction','hour_23_mean_direction',
                        'hour_0_skew_direction', 'hour_1_skew_direction','hour_2_skew_direction','hour_3_skew_direction','hour_4_skew_direction','hour_5_skew_direction','hour_6_skew_direction','hour_7_skew_direction','hour_8_skew_direction','hour_9_skew_direction','hour_10_skew_direction','hour_11_skew_direction','hour_12_skew_direction','hour_13_skew_direction','hour_14_skew_direction','hour_15_skew_direction','hour_16_skew_direction','hour_17_skew_direction','hour_18_skew_direction','hour_19_skew_direction','hour_20_skew_direction','hour_21_skew_direction','hour_22_skew_direction','hour_23_skew_direction',
                        'hour_0_kurt_direction', 'hour_1_kurt_direction','hour_2_kurt_direction','hour_3_kurt_direction','hour_4_kurt_direction','hour_5_kurt_direction','hour_6_kurt_direction','hour_7_kurt_direction','hour_8_kurt_direction','hour_9_kurt_direction','hour_10_kurt_direction','hour_11_kurt_direction','hour_12_kurt_direction','hour_13_kurt_direction','hour_14_kurt_direction','hour_15_kurt_direction','hour_16_kurt_direction','hour_17_kurt_direction','hour_18_kurt_direction','hour_19_kurt_direction','hour_20_kurt_direction','hour_21_kurt_direction','hour_22_kurt_direction','hour_23_kurt_direction',

                        'hour_period_0_mean_speed', 'hour_period_1_mean_speed','hour_period_2_mean_speed','hour_period_3_mean_speed',
                        'hour_period_0_std_speed', 'hour_period_1_std_speed','hour_period_2_std_speed','hour_period_3_std_speed',
                        'hour_period_0_mean_height', 'hour_period_1_mean_height','hour_period_2_mean_height','hour_period_3_mean_height',
                        'hour_period_0_std_height', 'hour_period_1_std_height','hour_period_2_std_height','hour_period_3_std_height',
                        'hour_period_0_mean_direc', 'hour_period_1_mean_direc','hour_period_2_mean_direc','hour_period_3_mean_direc',
                        'hour_period_0_skew_direc', 'hour_period_1_skew_direc','hour_period_2_skew_direc','hour_period_3_skew_direc',
                        'hour_period_0_kurt_direc', 'hour_period_1_kurt_direc','hour_period_2_kurt_direc','hour_period_3_kurt_direc']

        # callstate_feature_col = [
                                # 'callstate_0_mean_speed', 'callstate_1_mean_speed', 'callstate_2_mean_speed',
                                #  'callstate_3_mean_speed', 'callstate_4_mean_speed',
                                #  'callstate_0_std_speed', 'callstate_1_std_speed', 'callstate_2_std_speed',
                                #  'callstate_3_std_speed', 'callstate_4_std_speed',
                                #  'driver_calllstate_0_per', 'driver_calllstate_1_per', 'driver_calllstate_2_per',
                                #  'driver_calllstate_3_per', 'driver_calllstate_4_per'
                                #  ]

        # user_base_feature = ['user_mean_speed', 'user_max_speed', 'user_std_speed',
        #                      'user_mean_height', 'user_std_height', 'user_skew_height', 'user_kurt_height',
        #                      'user_mean_direc', 'user_skew_direc', 'user_kurt_direc']
        user_base_feature = ['user_mean_speed', 'user_max_speed', 'user_std_speed','user_skew_speed']

        lat_lon_feature = ['user_max_lon', 'user_min_lon', 'user_max_lat', 'user_min_lat',
                            'user_lon_ratio', 'user_lat_ratio', 'user_distance',
                            'user_max_geo_code', 'geo_code_max_per', 'user_loc_entropy','user_loc_num']
        all_feature = time_feature_col  + user_base_feature + lat_lon_feature

        feature = []
        for uid in data['TERMINALNO'].unique():
            hour_fea = data.loc[data['TERMINALNO'] == uid]
            feature.append(self.hour_fea(hour_fea))
        del data
        feature = pd.DataFrame(feature, columns=all_feature, dtype=np.float32)
        feature = feature.fillna(-1)
        return feature


    def callstate_fea(self, data):
        """
        用户状态特征
        """
        data = data.loc[data['SPEED'] >= 0]
        data = data.drop_duplicates()
        feature = []
        record_num = data.shape[0]

        callstate_speed_group = data['SPEED'].groupby(data['CALLSTATE'])
        # 各状态速度均值
        callstate_mean_speed = callstate_speed_group.mean()
        driver_callstate_speed = np.zeros(5, dtype=np.float32)
        driver_callstate_speed[callstate_mean_speed.index] = callstate_mean_speed
        feature.extend(driver_callstate_speed)
        del callstate_mean_speed

        # 各状态速度标准差
        callstate_std_speed = callstate_speed_group.std()
        driver_callstate_speed = np.zeros(5, dtype=np.float32)
        driver_callstate_speed[callstate_std_speed.index] = callstate_std_speed
        feature.extend(driver_callstate_speed)
        del callstate_std_speed

        # [5] 状态特征
        calllstate_per = data['CALLSTATE'].value_counts() / record_num
        driver_calllstate_per = np.zeros(5, dtype=np.float32)
        driver_calllstate_per[calllstate_per.index] = calllstate_per
        feature.extend(driver_calllstate_per)
        del calllstate_per

        return feature

    def user_callstatus_feature(self, data):

        callstate_feature_col = ['callstate_0_mean_speed','callstate_1_mean_speed','callstate_2_mean_speed','callstate_3_mean_speed','callstate_4_mean_speed',
                        'callstate_0_std_speed','callstate_1_std_speed','callstate_2_std_speed','callstate_3_std_speed','callstate_4_std_speed',
                        'driver_calllstate_0_per','driver_calllstate_1_per','driver_calllstate_2_per','driver_calllstate_3_per','driver_calllstate_4_per'
                        ]
        callstate_feature = []
        for uid in data['TERMINALNO'].unique():
            hour_fea = data.loc[data['TERMINALNO'] == uid]
            callstate_feature.append(self.callstate_fea(hour_fea))
        del data
        callstate_feature = pd.DataFrame(callstate_feature, columns=callstate_feature_col, dtype=np.float32)
        callstate_feature = callstate_feature.fillna(-1)
        return callstate_feature

    def user_direc_feature(self, data):
        """
        用户方向特征
        """
        data = data.loc[data['SPEED'] >= 0]
        data = data.drop_duplicates()
        features = []
        # [1] 行程量统计量
        # 总的行程记录数量
        record_num = data.shape[0]

        # [2] 速度特征
        speed_mean = data['SPEED'].mean()
        speed_max = data['SPEED'].max()
        speed_std = data['SPEED'].std()
        speed_median = data['SPEED'].median()
        features.extend([speed_mean, speed_max, speed_std, speed_median])

        # [3] 方向特征
        unknow_direc = (data['DIRECTION'] < 0).sum() / record_num
        features.append(unknow_direc)
        return  features


    def user_height_feature(self, data):
        data = data.loc[data['SPEED'] >= 0]
        data = data.drop_duplicates()
        features = []

        # [4]海拔特征
        height_mean = data['HEIGHT'].mean()
        height_max = data['HEIGHT'].max()
        height_min = data['HEIGHT'].min()
        height_std = data['HEIGHT'].std()
        height_median = data['HEIGHT'].median()
        features.extend([height_mean, height_max, height_min, height_std, height_median])

        return features

    def user_lon_lat_feature(self, data):
        data = data.loc[data['SPEED'] >= 0]
        data = data.drop_duplicates()
        features = []

        # 经纬度特征
        max_lon = data['LONGITUDE'].max()
        min_lon = data['LONGITUDE'].min()
        max_lat = data['LATITUDE'].max()
        min_lat = data['LATITUDE'].min()
        time_dur = (data['TIME'].max() - data['TIME'].min()) / 3600.0 + 1.0
        lon_ratio = (max_lon - min_lon) / time_dur
        lat_ratio = (max_lat - min_lat) / time_dur
        term = data.sort_values(by='TIME')
        startlong = term.iloc[0]['LONGITUDE']
        startlat = term.iloc[0]['LATITUDE']
        dis_start = self.haversine1(startlong, startlat, 113.9177317, 22.54334333)  # 距离某一点的距离

        # 将经纬度取整，拼接起来作为地块编码
        term['geo_code'] = term[['LONGITUDE', 'LATITUDE']].apply(lambda p: int(p[0]) * 100 + int(p[1]), axis=1)
        geo_sta = term['geo_code'].value_counts()
        loc_most = geo_sta.index[0]
        geo_sta = geo_sta / term.shape[0]
        loc_most_freq = geo_sta.iloc[0]
        loc_entropy = ((-1) * geo_sta * np.log2(geo_sta)).sum()
        loc_num = len(geo_sta)

        features.extend(
            [max_lon, min_lon, max_lat, min_lat, lon_ratio, lat_ratio, dis_start, loc_most, loc_most_freq, loc_entropy,
             loc_num])

        return features



time_feature = ['record_num',
'max_hour_num','min_hour_mun','driver_hour_mean_num','driver_hour_std_num','driver_hour_max_num','driver_hour_min_num',
'max_day_num','min_day_mun','driver_day_mean_num','driver_day_std_num','driver_day_max_num','driver_day_min_num',
'max_weekday_num','min_weekday_mun','driver_weekday_mean_num','driver_weekday_std_num','driver_weekday_max_num','driver_weekday_min_num',
'max_month_num','min_month_mun','driver_month_mean_num','driver_month_std_num','driver_month_max_num','driver_month_min_num',
'max_hour_period_num','min_hour_period_mun','driver_hour_period_mean_num','driver_hour_period_std_num','driver_hour_period_max_num','driver_hour_period_min_num',

'hour_0_trip', 'hour_1_trip','hour_2_trip','hour_3_trip','hour_4_trip','hour_5_trip','hour_6_trip','hour_7_trip','hour_8_trip'
,'hour_9_trip','hour_10_trip','hour_11_trip','hour_12_trip','hour_13_trip','hour_14_trip','hour_15_trip','hour_16_trip','hour_17_trip','hour_18_trip','hour_19_trip','hour_20_trip','hour_21_trip','hour_22_trip','hour_23_trip',

'hour_0_mean_speed', 'hour_1_mean_speed','hour_2_mean_speed','hour_3_mean_speed','hour_4_mean_speed','hour_5_mean_speed','hour_6_mean_speed','hour_7_mean_speed','hour_8_mean_speed','hour_9_mean_speed','hour_10_mean_speed','hour_11_mean_speed','hour_12_mean_speed','hour_13_mean_speed','hour_14_mean_speed','hour_15_mean_speed','hour_16_mean_speed','hour_17_mean_speed','hour_18_mean_speed','hour_19_mean_speed','hour_20_mean_speed','hour_21_mean_speed','hour_22_mean_speed','hour_23_mean_speed',

'hour_0_std_speed', 'hour_1_std_speed','hour_2_std_speed','hour_3_std_speed','hour_4_std_speed','hour_5_std_speed','hour_6_std_speed','hour_7_std_speed','hour_8_std_speed','hour_9_std_speed','hour_10_std_speed','hour_11_std_speed','hour_12_std_speed','hour_13_std_speed','hour_14_std_speed','hour_15_std_speed','hour_16_std_speed','hour_17_std_speed','hour_18_std_speed','hour_19_std_speed','hour_20_std_speed','hour_21_std_speed','hour_22_std_speed','hour_23_std_speed',

'hour_0_mean_height', 'hour_1_mean_height','hour_2_mean_height','hour_3_mean_height','hour_4_mean_height','hour_5_mean_height','hour_6_mean_height','hour_7_mean_height','hour_8_mean_height','hour_9_mean_height','hour_10_mean_height','hour_11_mean_height','hour_12_mean_height','hour_13_mean_height','hour_14_mean_height','hour_15_mean_height','hour_16_mean_height','hour_17_mean_height','hour_18_mean_height','hour_19_mean_height','hour_20_mean_height','hour_21_mean_height','hour_22_mean_height','hour_23_mean_height',
'hour_0_std_height', 'hour_1_std_height','hour_2_std_height','hour_3_std_height','hour_4_std_height','hour_5_std_height','hour_6_std_height','hour_7_std_height','hour_8_std_height','hour_9_std_height','hour_10_std_height','hour_11_std_height','hour_12_std_height','hour_13_std_height','hour_14_std_height','hour_15_std_height','hour_16_std_height','hour_17_std_height','hour_18_std_height','hour_19_std_height','hour_20_std_height','hour_21_std_height','hour_22_std_height','hour_23_std_height',
'hour_0_skew_height', 'hour_1_skew_height','hour_2_skew_height','hour_3_skew_height','hour_4_skew_height','hour_5_skew_height','hour_6_skew_height','hour_7_skew_height','hour_8_skew_height','hour_9_skew_height','hour_10_skew_height','hour_11_skew_height','hour_12_skew_height','hour_13_skew_height','hour_14_skew_height','hour_15_skew_height','hour_16_skew_height','hour_17_skew_height','hour_18_skew_height','hour_19_skew_height','hour_20_skew_height','hour_21_skew_height','hour_22_skew_height','hour_23_skew_height',
'hour_0_kurt_height', 'hour_1_kurt_height','hour_2_kurt_height','hour_3_kurt_height','hour_4_kurt_height','hour_5_kurt_height','hour_6_kurt_height','hour_7_kurt_height','hour_8_kurt_height','hour_9_kurt_height','hour_10_kurt_height','hour_11_kurt_height','hour_12_kurt_height','hour_13_kurt_height','hour_14_kurt_height','hour_15_kurt_height','hour_16_kurt_height','hour_17_kurt_height','hour_18_kurt_height','hour_19_kurt_height','hour_20_kurt_height','hour_21_kurt_height','hour_22_kurt_height','hour_23_kurt_height',

'hour_0_mean_direction', 'hour_1_mean_direction','hour_2_mean_direction','hour_3_mean_direction','hour_4_mean_direction','hour_5_mean_direction','hour_6_mean_direction','hour_7_mean_direction','hour_8_mean_direction','hour_9_mean_direction','hour_10_mean_direction','hour_11_mean_direction','hour_12_mean_direction','hour_13_mean_direction','hour_14_mean_direction','hour_15_mean_direction','hour_16_mean_direction','hour_17_mean_direction','hour_18_mean_direction','hour_19_mean_direction','hour_20_mean_direction','hour_21_mean_direction','hour_22_mean_direction','hour_23_mean_direction',
'hour_0_skew_direction', 'hour_1_skew_direction','hour_2_skew_direction','hour_3_skew_direction','hour_4_skew_direction','hour_5_skew_direction','hour_6_skew_direction','hour_7_skew_direction','hour_8_skew_direction','hour_9_skew_direction','hour_10_skew_direction','hour_11_skew_direction','hour_12_skew_direction','hour_13_skew_direction','hour_14_skew_direction','hour_15_skew_direction','hour_16_skew_direction','hour_17_skew_direction','hour_18_skew_direction','hour_19_skew_direction','hour_20_skew_direction','hour_21_skew_direction','hour_22_skew_direction','hour_23_skew_direction',
'hour_0_kurt_direction', 'hour_1_kurt_direction','hour_2_kurt_direction','hour_3_kurt_direction','hour_4_kurt_direction','hour_5_kurt_direction','hour_6_kurt_direction','hour_7_kurt_direction','hour_8_kurt_direction','hour_9_kurt_direction','hour_10_kurt_direction','hour_11_kurt_direction','hour_12_kurt_direction','hour_13_kurt_direction','hour_14_kurt_direction','hour_15_kurt_direction','hour_16_kurt_direction','hour_17_kurt_direction','hour_18_kurt_direction','hour_19_kurt_direction','hour_20_kurt_direction','hour_21_kurt_direction','hour_22_kurt_direction','hour_23_kurt_direction',

'hour_period_0_mean_speed', 'hour_period_1_mean_speed','hour_period_2_mean_speed','hour_period_3_mean_speed',
'hour_period_0_std_speed', 'hour_period_1_std_speed','hour_period_2_std_speed','hour_period_3_std_speed',
'hour_period_0_mean_height', 'hour_period_1_mean_height','hour_period_2_mean_height','hour_period_3_mean_height',
'hour_period_0_std_height', 'hour_period_1_std_height','hour_period_2_std_height','hour_period_3_std_height',
'hour_period_0_mean_direc', 'hour_period_1_mean_direc','hour_period_2_mean_direc','hour_period_3_mean_direc',
'hour_period_0_skew_direc', 'hour_period_1_skew_direc','hour_period_2_skew_direc','hour_period_3_skew_direc',
'hour_period_0_kurt_direc', 'hour_period_1_kurt_direc','hour_period_2_kurt_direc','hour_period_3_kurt_direc']


holiday_data = ['2016-09-15', '2016-09-16', '2016-09-17', '2016-10-01', '2016-10-02', '2016-10-03'
    , '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07', '2016-12-24', '2016-12-25'
    , '2016-12-03', '2016-12-31', '2017-01-01', '2016-07-02', '2016-07-03', '2016-07-09', '2016-07-10',
                '2016-07-16', '2016-07-17',
                '2016-07-23''2016-07-24', '2016-07-30', '2016-07-31',
                '2016-08-06', '2016-08-07''2016-08-13''2016-08-14', '2016-08-20', '2016-08-21',
                '2016-08-27', '2016-08-28',
                '2016-09-03', '2016-09-04', '2016-09-10', '2016-09-11', '2016-09-04', '2016-09-24'
    , '2016-09-25', '2016-10-15', '2016-10-16', '2016-10-22', '2016-10-23', '2016-10-29'
    , '2016-10-30', '2016-11-05', '2016-11-06', '2016-11-12', '2016-11-13', '2016-11-19'
    , '2016-11-20', '2016-11-26', '2016-11-27', '2016-12-03', '2016-12-04', '2016-12-10'
    , '2016-12-11', '2016-12-17', '2016-12-18']
