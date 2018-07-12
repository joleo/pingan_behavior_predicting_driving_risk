#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @createTime    :
# @author  :
from multiprocessing import Manager
from multiprocessing.synchronize import Lock
from multiprocessing import Process


class MulProcessHelper(object):
    """
    多进程辅助类
    """
    __manager = Manager()

    def __init__(self):
        self.__process_data_list = []
        self.__dct_share = None


    def subscribe_api(self, api_fcn, param=None, data_key="api_fcn"):
        """
        :type  m_p_h MulProcessHelper
        :param api_fcn:
        :param param:
        :param data_key:
        :param m_p_h:
        :return:
        """
        process_data = ProcessData(api_fcn, param, data_key)
        self.add_process_data_list(process_data)

    def add_process_data_list(self, process_data):
        """
        添加用于进程处理的方法队列
        :type process_data ProcessData
        :param process_data:
        :return:
        """
        self.__process_data_list.append(process_data)

    def flush_process(self, lock):
        """
        运行待处理的方法队列
        :type lock Lock
        :return 返回一个dict
        """
        # 覆盖上期变量
        self.__dct_share = self.__manager.Value('tmp', {})  # 进程共享变量

        p_list = []  # 进程列表
        i = 1
        p_size = len(self.__process_data_list)
        for process_data in self.__process_data_list:
            # 创建进程
            p = Process(target=self.__one_process, args=(process_data, lock))
            p.daemon = True
            p_list.append(p)
            # 间歇执行进程
            if i % 30 == 0 or i == p_size:  # 20页处理一次， 最后一页处理剩余
                for p in p_list:
                    p.start()
                for p in p_list:
                    p.join()  # 等待进程结束
                p_list = []  # 清空进程列表
            i += 1
        # end for

        self.__process_data_list = []  # 清空订阅

        return self.__dct_share.value

    def __run_process(self, lock):
        """
        运行待处理的方法队列
        """
        p_list = []  # 进程列表
        i = 1
        p_size = len(self.__process_data_list)
        for process_data in self.__process_data_list:
            # 创建进程
            p = Process(target=self.__one_process, args=(process_data, lock))
            p.daemon = True
            p_list.append(p)
            # 间歇执行进程
            if i % 20 == 0 or i == p_size:  # 20页处理一次， 最后一页处理剩余
                for p in p_list:
                    p.start()
                for p in p_list:
                    p.join()  # 等待进程结束
                p_list = []  # 清空进程列表
            i += 1

    def __one_process(self, process_data, lock):
        """
        处理进程
        :param process_data: 方法和参数集等
        :param lock: 保护锁
        """
        fcn = process_data.fcn
        params = process_data.params
        data_key = process_data.data_key
        data = fcn(params)
        with lock:
            temp_dct = dict(self.__dct_share.value)
            if data_key not in temp_dct:
                temp_dct[data_key] = []
            temp_dct[data_key].append(data)
            self.__dct_share.value = temp_dct


class ProcessData(object):
    """
    用于进程处理的的数据
    """

    def __init__(self, fcn, params, data_key):
        self.fcn = fcn  # 方法
        self.params = params  # 参数
        self.data_key = data_key  # 存储到进程共享变量中的名字