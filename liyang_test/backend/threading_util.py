#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# @Time    : 2018/4/4 22:04
# @Author  : liujiantao
# @Site    : 
# @File    : threading_util.py
# @Software: PyCharm
import threading
import time

def func1(ret_num):
  print ("func1 ret:%d" % ret_num)
  return ret_num
def func2(ret_num):
  print ("func2 ret:%d" % ret_num)
  return ret_num
def func3(ret_num):
  print ("func3 ret:%d"%ret_num)
  return ret_num

class ThreadingUtil(object): 
    def __init__(self, func_list=None):
        # 所有线程函数的返回值汇总，如果最后为0，说明全部成功
        self.data_list = []
        self.func_list = func_list
        self.threads = []

    def set_thread_func_list(self, func_list):
        """
         @note: func_list是一个list，每个元素是一个dict，有func和args两个参数
        :param func_list:
        :return:
        """
        self.func_list = func_list

    def trace_func(self, func, *args, **kwargs):
        """
        @note:替代profile_func，新的跟踪线程返回值的函数，对真正执行的线程函数包一次函数，以获取返回值
        """
        ret = func(*args, **kwargs)
        self.data_list.extend(ret)

    def start(self):
        """
        @note: 启动多线程执行，并阻塞到结束
        """
        self.threads = [] # 线程列表
        i = 1
        p_size = len(self.func_list)
        for func_dict in self.func_list:
            if func_dict["args"]:
                new_arg_list = []
                new_arg_list.append(func_dict["func"])
                for arg in func_dict["args"]:
                    new_arg_list.append(arg)
                new_arg_tuple = tuple(new_arg_list)
                t = threading.Thread(target=self.trace_func, args=new_arg_tuple)
            else:
                t = threading.Thread(target=self.trace_func, args=(func_dict["func"],))
            self.threads.append(t)
            # 间歇执行进程
            if i % 5 == 0 or i == p_size:  # 4页处理一次， 最后一页处理剩余
                for thread_obj in self.threads:
                    thread_obj.start()
                for thread_obj in self.threads:
                    thread_obj.join() # 等待线程结束
                self.threads = []
            i += 1
        # if thread_obj.if_stop == True:
        #     self.progress_threads.remove(threading)


if __name__ == '__main__':
    print(111)
    # t1 = threading.Thread(target=Worker1, args=("Hello",))
    # t2 = threading.Thread(target=Worker2, args=("Everyone",))
    # t1.start()
    # t2.start()
    mt = ThreadingUtil()
    g_func_list = []
    g_func_list.append({"func": func1, "args": (1,)})
    g_func_list.append({"func": func2, "args": (2,)})
    g_func_list.append({"func": func3, "args": (3,)})
    g_func_list.append({"func": func3, "args": (3,)})
    g_func_list.append({"func": func3, "args": (3,)})
    mt.set_thread_func_list(g_func_list)
    mt.start()
    print(mt.data_list)