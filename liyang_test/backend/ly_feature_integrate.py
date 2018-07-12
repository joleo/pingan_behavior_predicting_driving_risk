# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     lingyang_feature_integrate
   Description :
   Author :       liyang
   date：          2018/5/7 0007
-------------------------------------------------
   Change Activity:
                   2018/5/7 0007:
-------------------------------------------------
"""
__author__ = 'liyang'


from data_helper import DataHelper
from ly_feature_etl1 import *
from threading_util import ThreadingUtil
from config import *

class LYFeatureIntegrate(object):
    def __init__(self):
        self.d_h = DataHelper()

    def get_all_features(self, path=path_train01):
        start = time.time()
        data = self.d_h.get_data(path)
        ft_Liyang = LYFeatureExtraction1(self.d_h, data)
        user_Y_list = self.d_h.get_user_Y_list(data)
        min_Y = data[data['Y'] != 0]['Y'].min()
        max_Y = data[data['Y'] != 0]['Y'].max()

        mt = ThreadingUtil()
        g_func_list = []
        # 用户行程占比
        # g_func_list.append({"func": ft_Liyang.driver_time_feature, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.get_feat, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.driver_base_feature, "args": (data,)})
        # g_func_list.append({"func": ft_Liyang.user_callstate_feature, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_direction__rate, "args": (data,)})
        # g_func_list.append({"func": ft_Liyang.user_height_rate, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_speed_rate, "args": (data,)})

        mt.set_thread_func_list(g_func_list)
        mt.start()

        all_features_list = [[row[col] for row in mt.data_list] for col in range(len(mt.data_list[0]))]

        self.d_h.print_str += " get_train_features cost time: " + str(time.time() - start) + " "
        return all_features_list, user_Y_list,min_Y,max_Y

    def get_test_features02(self, path=path_test01):
        """

        :param path:
        :return:
        """
        start = time.time()
        data = self.d_h.get_test_data(path)
        userid_list = self.d_h.get_userlist(data)

        ft_Liyang = LYFeatureExtraction1(self.d_h, data)
        mt = ThreadingUtil()
        g_func_list = []
        # g_func_list.append({"func": ft_Liyang.driver_time_feature, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.get_feat, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.driver_base_feature, "args": (data,)})
        # g_func_list.append({"func": ft_Liyang.user_callstate_feature, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_direction__rate, "args": (data,)})
        # g_func_list.append({"func": ft_Liyang.user_height_rate, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_speed_rate, "args": (data,)})
        mt.set_thread_func_list(g_func_list)
        mt.start()

        test_features = [[row[col] for row in mt.data_list] for col in range(len(mt.data_list[0]))]
        self.d_h.print_str += " get_test_features cost time: " + str(time.time() - start) + " "
        return userid_list,test_features

if __name__ == '__main__':
    lyetl = LYFeatureIntegrate()
    all_features_list,user_Y_list,min_Y,max_Y = lyetl.get_all_features()
    print(max_Y)