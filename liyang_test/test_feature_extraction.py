# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_feature_extraction
   Description :
   Author :       Administrator
   date：          2018/4/12 0012
-------------------------------------------------
   Change Activity:
                   2018/4/12 0012:
-------------------------------------------------
"""
__author__ = 'Administrator'

from unittest import TestCase
from data_helper import DataHelper
from data_util import DataUtil
from feature_extraction import *
import warnings
warnings.filterwarnings('ignore')

d_h = DataHelper()
d_t = DataUtil()
path = '../data/dm/train.csv'
data = pd.read_csv(path)

fe = LYFeatureExtraction()

class TestFeature(TestCase):

    def test_1(self):
        test = fe.test(data)
        print(test.head())
        assert True

    def test_user_geo_stat(self):
        user_geo_stat = fe.user_geo_stat(data)
        print(user_geo_stat.head())
        assert True

    def test_user_Y(self):
        user_Y = fe.get_user_Y(data)
        print(user_Y)
        assert True

    def test_user_driver_time(self):
        """

        """
        user_driver_time = fe.user_driver_time(data)
        print(user_driver_time.head())
        assert True

    def test_user_night_stat(self):
        """
        Ran 1 test in 1.440s
        """
        user_night_stat = fe.user_night_stat(data)
        # print(user_night_stat.columns)
        print(user_night_stat.info())
        assert True

    def test_get_distance(self):
        get_distance = fe.get_distance(data)
        print(get_distance.head())
        assert True

    def test_get_similar_position(self):
        get_similar_position = fe.get_similar_position(data)
        print(get_similar_position)
        assert True

    def test_user_driver_stat(self):
        """

        """
        user_driver_stat = fe.user_driver_stat(data)
        print(user_driver_stat.head())
        # print(test)
        assert True

    def test_user_callstate_stat(self):
        """
        Ran 1 test in 0.585s
        """
        user_callstate_stat = fe.user_callstate_stat(data)
        print(user_callstate_stat.head())
        # print(test)
        assert True

    def test_user_direction__stat(self):
        """
        方向特征
        """
        user_direction__stat = fe.user_direction__stat(data)
        print(user_direction__stat.columns)
        assert True

    def test_user_height_stat(self):
        """
        海拔特征
        """
        user_height_stat = fe.user_height_stat(data)
        print(user_height_stat.columns)
        assert True

    def test_user_speed_stat(self):
        """
        速度特征统计
        """
        user_speed_stat = fe.user_speed_stat(data)
        print(user_speed_stat.columns)
        assert True


