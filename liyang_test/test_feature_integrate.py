# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_feature_integer
   Description :
   Author :       Administrator
   date：          2018/5/5 0005
-------------------------------------------------
   Change Activity:
                   2018/5/5 0005:
-------------------------------------------------
"""
__author__ = 'Administrator'

import warnings

from unittest import TestCase
from data_helper import DataHelper
from data_util import DataUtil
from feature_integrate import *
d_h = DataHelper()
d_t = DataUtil()
path = '../data/dm/train.csv'
data = pd.read_csv(path)

fi = FeatureIntegrate()

class TestFeature(TestCase):

    def test_train_feature(self):
        """
        Ran 1 test in 3.251s
        """
        train_feature = fi.train_feature_integrate(data)
        print(train_feature.columns)
        assert True

    def test_test_feature(self):
        """

        """
        test_feature = fi.test_feature_integrate(data)
        print(test_feature.columns)
        assert True
