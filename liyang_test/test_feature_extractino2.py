# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_feature_extractino2
   Description :
   Author :       Administrator
   date：          2018/5/20 0020
-------------------------------------------------
   Change Activity:
                   2018/5/20 0020:
-------------------------------------------------
"""
__author__ = 'Administrator'
from unittest import TestCase
from data_helper import DataHelper
from data_util import DataUtil
from feature_extraction2 import *
import warnings
warnings.filterwarnings('ignore')

d_h = DataHelper()
d_t = DataUtil()
path = '../data/dm/train.csv'
data = pd.read_csv(path)

fe = LYFeatureExtraction()

class TestFeature(TestCase):

    def test_user_hour_feature(self):
        user_hour_feature = fe.user_hour_feature(data)
        print(user_hour_feature.head(2).T)
        assert True

    def test_user_callstatus_feature(self):
        user_callstatus_feature = fe.user_callstatus_feature(data)
        print(user_callstatus_feature.head(2).T)
        assert True


    def test_gen_time_feature(self):
        gen_time_feature = fe.gen_time_feature(data)
        print(gen_time_feature.head(2))
        assert True