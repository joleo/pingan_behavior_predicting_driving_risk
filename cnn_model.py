# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cnn_model
   Description :
   Author :       joleo
   date：          2018/5/9 0009
-------------------------------------------------
   Change Activity:
                   2018/5/9 0009:
-------------------------------------------------
"""
__author__ = 'joleo'

import time
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
from feature_integrate import *
from config import *

fi = FeatureIntegrate()

start = time.time()
start_time = time.time()

# 变厚矩阵
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积处理 变厚过程
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1] x_movement、y_movement就是步长
    # Must have strides[0] = strides[3] = 1 padding='SAME'表示卷积后长宽不变
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


def fit(pre_features_list, act_Y, test_features, start_time):
   train_x_disorder = pre_features_list
   train_y_disorder = act_Y.reshape(-1, 1)
   # train_x_disorder_3 = train_x_disorder[:, 3:6]
   # train_x_disorder = np.column_stack([train_x_disorder, train_x_disorder_3])  # 随意给x增加了1列，x变为16列，可以reshape为4*4矩阵了 没啥用，就是凑个正方形
   # test_features_3 = test_features[:, 3:6]
   # test_features = np.column_stack([test_features, test_features_3])  # 随意给x增加了1列，x变为16列，可以reshape为4*4矩阵了 没啥用，就是凑个正方形

   print(test_features.shape)
   # 随机挑选
   # train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x, y,  train_size=0.8, random_state=33)
   # # 数据标准化
   ss_x = preprocessing.StandardScaler()
   train_x_disorder = ss_x.fit_transform(train_x_disorder)
   test_features = ss_x.transform(test_features)
   # train_x_disorder = normalize_cols(train_x_disorder)
   # test_features = normalize_cols(test_features)

   # define placeholder for inputs to network
   xs = tf.placeholder(tf.float32, [None, 36])  # 原始数据的维度：16
   ys = tf.placeholder(tf.float32, [None, 1])  # 输出数据为维度：1

   keep_prob = tf.placeholder(tf.float32)  # dropout的比例

   x_image = tf.reshape(xs, [-1, 6, 6, 1])  # 原始数据16变成二维图片4*4

   # 初始化算法模型
   ## conv1 layer ##第一卷积层
   W_conv1 = weight_variable([2, 2, 1, 72])  # patch 2x2, in size 1, out size 32,每个像素变成32个像素，就是变厚的过程
   b_conv1 = bias_variable([72])
   h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 2x2x32，长宽不变，高度为32的三维图像
   # h_pool1 = max_pool_2x2(h_conv1)     # output size 2x2x32 长宽缩小一倍

   ## conv2 layer ##第二卷积层
   W_conv2 = weight_variable([2, 2, 72, 144])  # patch 2x2, in size 32, out size 64
   b_conv2 = bias_variable([144])
   h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  # 输入第一层的处理结果 输出shape 4*4*64

   # 池化层
   # max_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
   #                            strides=[1, 2, 2, 1], padding='SAME')
   #
   # # 转换为1*n层，进行平整化
   # final_conv_shape = max_pool2.get_shape().as_list()
   # final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
   # flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

   ## fc1 layer ##  full connection 全连接层
   W_fc1 = weight_variable([6 * 6 * 144, 648])  # 4x4 ，高度为64的三维图片，然后把它拉成512长的一维数组
   b_fc1 = bias_variable([648])
   h_pool2_flat = tf.reshape(h_conv2, [-1, 6 * 6 * 144])  # 把4*4，高度为64的三维图片拉成一维数组 降维处理
   h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 把数组中扔掉比例为keep_prob的元素
   ## fc2 layer ## full connection
   W_fc2 = weight_variable([648, 1])  # 512长的一维数组压缩为长度为1的数组
   b_fc2 = bias_variable([1])  # 偏置

   # 最后的计算结果
   prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
   # prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
   # 计算 predition与y 差距 所用方法很简单就是用 suare()平方,sum()求和,mean()平均值
   cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
   # 0.01学习效率,minimize(loss)减小loss误差
   train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)

   sess = tf.Session()
   sess.run(tf.global_variables_initializer())
   # 训练200次
   for i in range(3000):
      sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 0.7})
      loss = sess.run(cross_entropy,feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 0.7})
      print(i, 'the loss =', loss)  # 输出loss值
      if (time.time() - start_time) > 20000:
         break
      if loss < 4:
         break
   prediction_value = sess.run(prediction, feed_dict={xs: test_features, keep_prob: 0.7})
   return prediction_value


# 为了后面复现结果，为numpy和tensorflow设置随机种子
seed = 1514
np.random.seed(seed)
tf.set_random_seed(seed)


# 载入数据
train_data = pd.read_csv(path_train01)
train_data = train_data.ix[:15000000, :]
test_data = pd.read_csv(path_test01)

train = fi.train_feature_integrate(train_data)
test = fi.test_feature_integrate(test_data)

# 36个特征
# feature = [x for x in train.columns if x not in ['TERMINALNO','Y','hour_count_max','night_count_max',
#                                                  'user_speed_skew', 'user_height_skew', 'direction_skew',
#                                                  ' hour_2_per','user_distance']]
feature = [x for x in train.columns if x not in ['TERMINALNO','Y','hour_count_max'
                                                 ,'hour_2_per','night_count_max','user_distance']]

print(feature)

print(train[feature].shape) # 33

train_x = train[feature].fillna(-1)
train_y = train['Y']
test_x = test[feature].fillna(-1)

train_x = train_x.values
train_y = train_y.values
test_x = test_x.values

prediction_value = fit(train_x, train_y, test_x, start_time)

# pre_result = {
#    "Id": [],
#    "Pred": []
# }
pred = []
result = pd.DataFrame(test['TERMINALNO'])
user_id = list(set(result['TERMINALNO'].tolist()))
# for i in range(len(user_id)):
#    user = user_id[i]
#    pre_result['Id'].append(user)
#    y1 = round(prediction_value[i][0], 9)
#    pre_result['Pred'].append(y1)
# df = pd.DataFrame(pre_result, columns=['Id', 'Pred'])
#
# df['Pred'].map(lambda x: abs(x))#map(lambda x: abs(x)/10 if x < 0 else x )
for i in range(len(user_id)):
   user = user_id[i]
   y1 = round(prediction_value[i][0], 9)
   pred.append(y1)
result['Pred'] = pred
result = result.rename(columns={'TERMINALNO':'Id'})
result.loc[:, 'Pred'] = result['Pred'].map(lambda x: abs(x)/10 if x < 0 else x )
# df = pd.DataFrame(pre_result, columns=['Id', 'Pred'])
result.to_csv(path_test_out01,header=True,index=False)

print("cost time: " + str(time.time() - start))