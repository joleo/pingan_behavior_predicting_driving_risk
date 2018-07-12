# -*- coding: utf-8 -*-
from data_helper import DataHelper
from ly_feature_etl2 import *
from threading_util import ThreadingUtil
from config import *

class LYFeatureIntegrate2(object):
    def __init__(self):
        self.d_h = DataHelper()

    def get_all_features(self, path=path_train01):
        start = time.time()
        data = self.d_h.get_data(path)
        fe = LYFeatureExtraction2(self.d_h, data)
        user_Y_list = self.d_h.get_user_Y_list(data)

        mt = ThreadingUtil()
        g_func_list = []

        g_func_list.append({"func": fe.user_driver_time, "args": (data,)})
        g_func_list.append({"func": fe.user_night_stat, "args": (data,)})
        g_func_list.append({"func": fe.user_driver_stat, "args": (data,)})
        g_func_list.append({"func": fe.get_distance, "args": (data,)})
        g_func_list.append({"func": fe.user_direction__stat, "args": (data,)})
        g_func_list.append({"func": fe.user_height_stat, "args": (data,)})
        g_func_list.append({"func": fe.user_speed_stat, "args": (data,)})

        mt.set_thread_func_list(g_func_list)
        mt.start()

        all_features_list = [[row[col] for row in mt.data_list] for col in range(len(mt.data_list[0]))]

        self.d_h.print_str += " get_train_features cost time: " + str(time.time() - start) + " "
        return all_features_list, user_Y_list

    def get_test_features02(self, path=path_test01):
        """

        :param path:
        :return:
        """
        start = time.time()
        data = self.d_h.get_test_data(path)
        userid_list = self.d_h.get_userlist(data)

        ft_Liyang = LYFeatureExtraction2(self.d_h, data)
        mt = ThreadingUtil()
        g_func_list = []
        g_func_list.append({"func": ft_Liyang.user_driver_time, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_night_stat, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_driver_stat, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.get_distance, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_direction__stat, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_height_stat, "args": (data,)})
        g_func_list.append({"func": ft_Liyang.user_speed_stat, "args": (data,)})
        mt.set_thread_func_list(g_func_list)
        mt.start()

        test_features = [[row[col] for row in mt.data_list] for col in range(len(mt.data_list[0]))]
        self.d_h.print_str += " get_test_features cost time: " + str(time.time() - start) + " "
        return userid_list,test_features

if __name__ == '__main__':
    lyetl = LYFeatureIntegrate2()
    # all_features_list,user_Y_list = lyetl.get_all_features()
    # print(all_features_list)

    userid_list, test_features = lyetl.get_test_features02()
    print(userid_list)

