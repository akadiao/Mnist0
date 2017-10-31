# Mnist0
＃＃＃＃＃＃＃＃＃＃备份程序


# #!/usr/bin/python
# # coding:utf-8

"""# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

    sess = tf.InteractiveSession()
    # Create a multilayer model.

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    # Do not apply softmax activation yet, see below.
    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}


    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()



def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
    parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
    parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# # MNIST入门
#
# import tensorflow as tf
#
# import input_data
# mnist = input_data.read_data_sets('Mnist_data', one_hot=True)
#
# # x不是一个特定的值，而是一个占位符
# # 能够输入任意数量的MNIST图像，每一张图展平成784维的向量
# x = tf.placeholder("float", [None, 784])
# #  一个Variable代表一个可修改的张量
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# # y=softmax(Wx+b)
# y = tf.nn.softmax(tf.matmul(x, W)+b)
# print y
# # 添加一个新的占位符用于输入正确值
# y_ = tf.placeholder("float", [None, 10])
# # 计算交叉熵
# # 用tf.reduce_sum 计算张量的所有元素的总和
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# # 以0.01的学习速率最小化交叉熵
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# # 初始化变量
# init = tf.initialize_all_variables()
# # 在一个Session里面启动模型
# sess = tf.Session()
# sess.run(init)
#
# # 让模型循环训练1000次
# for i in range(1000):
#     # 随机抓取训练数据中的100个批处理数据点
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     # 用这些数据点作为参数替换之前的占位符来运行train_step
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# # 检测预测是否与实际标签匹配,返回一组布尔值
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# # 把布尔值转换成浮点数，然后取平均值
# # [True, False, True, True]变成[1,0,1,1],平均后得0.75
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# # 计算所学习到的模型在测试数据集上面的正确率
# # result:0.9149
# print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


"""





"""



#深入MNIST


# import tensorflow as tf
# import input_data
# # 下载并读取数据
# mnist = input_data.read_data_sets('Mnist_data', one_hot=True)
# # 运行交互计算图
# sess = tf.InteractiveSession()

# # 构建Softmax 回归模型
#
# # 用占位符定义输入图片x与输出类别y_
# x = tf.placeholder("float", shape=[None, 784])
# y_ = tf.placeholder("float", shape=[None, 10])
#
# # 将权重W和偏置b定义为变量,并初始化为0向量
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# # 变量需要通过seesion初始化后才能在session中使用
# sess.run(tf.initialize_all_variables())
#
# # 类别预测与损失函数
#
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# # 为训练过程指定最小化误差用的损失函数(此处用交叉熵)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#
# # 训练模型
#
# # 用最速下降法让交叉熵下降,步长为0.01
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# # 每次加载50个训练样本,然后执行一次train_step,通过feed_dict将x和y_用训练训练数据替代
# for i in range(1000):
#     batch = mnist.train.next_batch(50)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
# # 评估模型
# # 用tf.equal来检测预测是与否真实标签匹配
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# # 将布尔值转换为浮点数来代表对错然后取平均值
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# print accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
#


"""









import tensorflow as tf
# import tensorflow.examples.tutorials.mnist.input_data as input_data
import input_data
# 下载并读取数据
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)
# 运行交互计算图
# sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 权重初始化
# 用一较小正数初始化偏置项以避免神经元节点输出恒为0
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积和池化
# 卷积步长为1边界用0填充
def conv2d(x, W):
    # 输入图像shape=[batch, in_height, in_width, in_channels] float32/float64
    # 卷积核  shape=[filter_height, filter_width, in_channels, out_channels]
    # strides 在图像每一维的步长
    # padding string类型的量”SAME”/”VALID”
    # 结果返回一个Tensor　即feature map
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 池化用2x2大小的模板做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# [一个batch的图片数量,宽,高,通道数]
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 第一层卷积
# 卷积的权重张量形状是[5, 5, 1, 32],前两个维度是patch的大小,接着是输入的通道数目,最后是输出的通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 第二层卷积
# 每个5x5的patch会得到64个特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# 密集连接层
# 加入一个有1024个神经元的全连接层将图片尺寸减小到7x7
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
# 添加一个softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 训练和评估模型
# 在feed_dict中加入额外的参数keep_prob来控制dropout比例
# 每100次迭代输出一次日志
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        # 训练过程中启用dropout
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 测试过程中关闭dropout
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print "test accuracy %g" % acc

sess.close()
