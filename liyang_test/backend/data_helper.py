#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# @Time    : 2018/3/31 22:54
# @Author  : liujiantao
# @Site    : 
# @File    : data_helper.py
# @Software: PyCharm
import datetime
import numpy as np
import json
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from data_util import DataUtil

dtypes = {
    'TERMINALNO': 'uint32',
    'USER_TRIP_ALL_CNT': 'float64',
    'TRIP_ID': 'float64',
    'LATITUDE': 'float64',
    'LONGITUDE': 'float64',
    'DIRECTION': 'float64',
    'HEIGHT': 'float64',
     'SPEED': 'float64',
    'CALLSTATE': 'float64',
    'Y':'float64'
    }

d_test_types = {
    'TERMINALNO': 'uint32',
    'USER_TRIP_ALL_CNT': 'float64',
    'TRIP_ID': 'float64',
    'LATITUDE': 'float64',
    'LONGITUDE': 'float64',
    'DIRECTION': 'float64',
    'HEIGHT': 'float64',
    'SPEED': 'float64',
    'CALLSTATE': 'float64'
    }

class DataHelper(object):
    """
    数据辅助类
    """
    print_str = ""

    def get_data(self, path_train):
        # data = pd.read_csv(path_train, dtype=dtypes,nrows =4914734)
        data = pd.read_csv(path_train, dtype=dtypes)
        # 随机抽取20%的测试集
        # X_train, X_test = train_test_split(data, test_size= 0.3, random_state=0)
        return data


    def get_test_data(self, path_test):
        data = pd.read_csv(path_test, dtype=d_test_types)
        # X_train, X_test = train_test_split(data, test_size=0.3, random_state=0)
        return data

    def get_userlist(self, data):
        return list(set(data['TERMINALNO'].tolist()))


    def get_user_Y_list(self, data):
        uy = sorted(list(data.groupby(['TERMINALNO', 'Y']).indices), key=lambda item: item[0])
        y = [col[1] for col in uy]
        return y

    @staticmethod
    def loadDataSet(path):
        dataMat = []
        labelMat = []
        data = pd.read_csv(path, dtype=dtypes)
        print("data len : " + str(len(dataMat)))
        return dataMat, labelMat

    @staticmethod
    def my_print(*lists):
        """
        如果参数列表最后一位为 False 就拒绝打印
        """

        print(DataUtil.decode(lists))

    @staticmethod
    def print_domain(obj):
        """
        领域对象打印
        """
        print(json.dumps(obj, ensure_ascii=False, indent=4, default=lambda x: x.__dict__))

    @staticmethod
    def equal(var1, var2):
        """
        验证两个浮点数 对象是否大致相等
        """
        return var1 - var2 < 0.00001

    @staticmethod
    def assert_equal(expected, actual, message=''):
        if expected != actual:
            assert expected == actual, '{} 期待值:{} 实际值{}'.format(message, expected,actual)

    @staticmethod
    def side_effect(reason_dict, default_value):
        """
        返回一个mock 方法
        :param reason_dict: 第一个输入值，输出值匹配字典表
        :param default_value: 未匹配的默认值
        :return: 用于设置 相关mock的inner_effect属性
        """

        def inner_effect(*arg):
            input_name = arg[0]
            if input_name in reason_dict.keys():
                return reason_dict[input_name]
            else:
                return default_value

        return inner_effect


    def __gini(self, y_true, y_pred):
        # check and get number of samples
        assert y_true.shape == y_pred.shape
        n_samples = y_true.shape[0]

        # sort rows on prediction column
        # (from largest to smallest)
        arr = np.array([y_true, y_pred]).transpose()
        true_order = arr[arr[:, 0].argsort()][::-1, 0]
        pred_order = arr[arr[:, 1].argsort()][::-1, 0]

        # get Lorenz curves
        L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
        L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
        L_ones = np.linspace(1 / n_samples, 1, n_samples)

        # get Gini coefficients (area between curves)
        G_true = np.sum(L_ones - L_true)
        G_pred = np.sum(L_ones - L_pred)
        if G_pred == 0.0 or G_true == 0.0:
            return 1.0
        # normalize to true Gini coefficient
        return G_pred * 1.0 / G_true

    @staticmethod
    def dcg_score(pred, label, k=5):
        order = np.argsort(pred)[::-1]
        y_true = np.take(label, order[:k])
        gain = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)

    @staticmethod
    def idcg_score(pred, label):
        order = np.argsort(pred)[::-1]
        y_true = np.take(label, order)
        gain = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)

    # 排序 rank score
    def ndcg_score(self, pred, label, k=None):
        if k == None:
            dcg_max = self.idcg_score(label, label)
            dcg_min = self.idcg_score(label, -label)
            assert dcg_max > dcg_min
            if not dcg_max:
                return 0.
            dcg = self.idcg_score(pred, label)
            return (dcg - dcg_min) / (dcg_max - dcg_min)
        dcg_max = self.dcg_score(label, label, k)
        dcg_min = self.dcg_score(label, -label, k)
        assert dcg_max > dcg_min
        if not dcg_max:
            return 0.
        dcg = self.dcg_score(pred, label, k)
        return (dcg - dcg_min) / (dcg_max - dcg_min)

    @staticmethod
    def gini(actual, pred):
        assert (len(actual) == len(pred))
        all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        total_losses = all[:, 0].sum()
        gini_sum = all[:, 0].cumsum().sum() / total_losses
        gini_sum -= (len(actual) + 1) / 2.
        return gini_sum / len(actual)

    def gini_normalized(self, actual, pred):
        return self.gini(actual, pred) / self.gini(actual, actual)

    @staticmethod
    def inc_eprem(sorted_value, premium, reparations):
        order = np.argsort(sorted_value)
        premium_sorted = np.take(premium, order)
        reparations_sorted = np.take(reparations, order)
        premium_cumsum = np.cumsum(premium_sorted)
        reparations_cumsum = np.cumsum(reparations_sorted)
        premium_sum = np.sum(premium)
        reparations_sum = np.sum(reparations)
        x = premium_cumsum / premium_sum
        y = reparations_cumsum / reparations_sum
        return x, y

    def pingan_gini(self, sorted_value, premium, reparations):
        x, y = self.inc_eprem(sorted_value, premium, reparations)
        ret = self.gini(x, y)
        return ret

    @staticmethod
    def idcg_mse(sorted_value, f):
        order = np.argsort(sorted_value)[::-1]
        f_order = np.take(f, order)
        f_rank = np.sort(f)[::-1]
        ret1 = np.sum((f_order - f_rank) ** 2 / len(f_order))
        ret2 = np.sqrt(ret1)
        return ret2

    def eval_accuracy_score(self, y_test, ytestPre):
        '''
        准确率\微召回率\调和平均数 计算
        :param y_test:
        :param ytestPre:
        :return:
        '''
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, ytestPre)
        print(u'准确率：' +str(100 * accuracy))
        from sklearn import metrics
        precision = metrics.precision_score(y_test, ytestPre, average='micro')  # 微平均，精确率
        print(u'微平均，精确率： ' +str(100 * precision))
        recall = metrics.recall_score(y_test, ytestPre, average='macro')
        print(u'微平均，召回率： ' +str(100 * recall))
        f1_score = metrics.f1_score(y_test, ytestPre, average='weighted')
        print(u'微平均，调和平均数： ' +str(100 * f1_score))

    def eval_classification_report(self, y_test, ytestPre,target_names):
        """
         分类报告
        :param y_test: 实际测试值
        :param ytestPre: 预测值
        :param target_names: 类别标签
        :return:
        """
        from sklearn.metrics import classification_report
        classification_report(y_test, ytestPre, target_names=target_names)

    def evaluate_function(self, clf, X_test, y_test,target_names):
        ytestPre = clf.predict(X_test)
        from sklearn.metrics import accuracy_score
        # accuracy = accuracy_score(y_test, ytestPre)
        # print(u'准确率： ' +str(100 * accuracy))
        # from sklearn import metrics
        # precision = metrics.precision_score(y_test, ytestPre, average='micro')  # 微平均，精确率
        # print(u'微平均，精确率： ' +str(100 * precision))
        # recall = metrics.recall_score(y_test, ytestPre, average='macro')
        # print(u'微平均，召回率： ' +str(100 * recall))
        # f1_score = metrics.f1_score(y_test, ytestPre, average='weighted')
        # print(u'微平均，调和平均数： ' +str(100 * f1_score))
        from sklearn.metrics import classification_report
        print(classification_report(y_test, ytestPre, target_names=target_names))

    def eval_cohen_kappa_score(self, y_test, y_Pre):
        """
        函数cohen_kappa_score计算了Cohen’s kappa估计。这意味着需要比较通过不同的人工标注（numan annotators）的标签，而非分类器中正确的类。
        kappa score是一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好（实际是随机标签）
        Kappa score可以用在二分类或多分类问题上，但不适用于多标签问题，以及超过两种标注的问题。
        :param y_test:
        :param y_Pre:
        :return:
        """
        from sklearn.metrics import cohen_kappa_score
        kappa_score = cohen_kappa_score(y_test, y_Pre)
        print(u'kappa score是一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好（实际是随机标签）： ' +str(100 * kappa_score))

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
        return timeStamp, timeArray  # 1381419600

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