#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# @Time    : 2018/3/30 23:26
# @Author  : liujiantao
# @Site    : 数据处理
# @File    : data_util.py
# @Software: PyCharm
import json
import re
import sys
import datetime
import time

class DataUtil(object):
    """
    数据处理工具
    """
    @staticmethod
    def split_data(X, y, test_size=0.33):
        '''
        train_test_split(train_data,train_target,test_size=0.4, random_state=0)
        train_data：所要划分的样本特征集
        train_target：所要划分的样本结果
        test_size：样本占比，如果是整数的话就是样本的数量
        random_state：是随机数的种子。
        随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，
        其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
        随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
        种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
        :param X:
        :param y:
        :return:
        '''
        from sklearn.model_selection import train_test_split
        # 随机抽取20%的测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def timeStampToStrTime(timeStamp):
        '''
        时间戳转时间
        :return:
        '''
        dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
        otherStyleTime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
        return otherStyleTime

    @staticmethod
    def StrTimeToTimeStamp(tss1):
        '''
        时间转时间戳
        :return:
        '''
        # 转为时间数组
        timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
        # timeArray可以调用tm_year等
        # 转为时间戳
        timeStamp = int(time.mktime(timeArray))
        return   timeStamp,timeArray # 1381419600

    @staticmethod
    def is_none(d):
        '''
        判断字符串是否为空包括‘’，NULL，,None
        :param d:
        :return:
        '''
        return (d is None or d == 'None' or
                d == '' or
                d == {} or
                d == [] or
                d == 'NULL' or
                d == 'null')

    @staticmethod
    def last_word_cut(text):
        """
        剪接最后一个字符
        """
        text = text[:len(text) - 1]
        return text

    @staticmethod
    def is_in_str(str_list, trg_str):
        is_find = False

        if trg_str:
            for s in str_list:
                if s in trg_str:
                    is_find = True
                    break
            # end for
        # end if

        return is_find

    @staticmethod
    def right_cut_by_word(text, cut_word):
        """
        右向剪断字符
        input: text= 'good/bye/oo', cut_word = 'bye'
        output: 'good/'
        """
        i = text.find(cut_word)
        if i != -1:
            text = text[0: i]

        return text

    @staticmethod
    def last_word_cut_num(text, cut_num):
        """
        剪接最后指定数量字符
        """
        text = text[:len(text) - cut_num]
        return text

    @staticmethod
    def get_replace_pattern(text_parts, context, str_to_replace=u''):
        """
        正则替换匹配值为空
        :param text_parts:
        :param context:
        :param str_to_replace:
        :return:
        """
        context = context.encode('utf-8').decode('utf-8')
        return re.sub(text_parts, str_to_replace, context)

    @staticmethod
    def decode(input_str):
        """
        中文解码
        """
        return json.dumps(input_str, ensure_ascii=False, indent=4, default=lambda x: str(x))

    @staticmethod
    def contain_var_in_string(containVar, stringVar):
        '''
        python判断字符串中包含某个字符的判断函数脚本
        :param containVar:查找包含的字符
        :param stringVar:所要查找的字符串
        :return:
        '''
        if isinstance(stringVar, str):
            if containVar in stringVar:
            # if stringVar.find(containVar) > -1:
                return True
            else:
                return False
        else:
            return False