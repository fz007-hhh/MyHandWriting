from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from work.Method import Method

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_datas=np.load('work/self_data_64_64_all.npy')
train_targets=np.load('work/self_target_64_64_all.npy')

# 测试源
x = tf.placeholder(tf.float32, [None, 4096])#设置图片大小64*64=4096px
# 目标结果，0-9，共19个数字
y_ = tf.placeholder(tf.float32, [None, 10])


def weight_variable(shape):#权重初始化
    # truncated_normal截断正态分布随机数
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):#偏置
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):#卷积
    # conv2d 在给定的输入input和过滤filter条件下计算2D卷积
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):#池化
    # 最大值方法池化，将大矩阵分成上下左右各部分，取每部分的最大数形成小矩阵
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])#权重张量形状[5, 5, 1, 32]
b_conv1 = bias_variable([32])#对应偏置量为[32]

# reshape 将n个2500*1的向量，转化为n个28*28的4维向量
x_image = tf.reshape(x,[-1, 64, 64, 1])#将x转化为一个4维向量，28*28表示宽高，1表示颜色通道，图片无颜色，若是有颜色的图片，则为3

# relu 把小于0的值置为0，大于0的值不变
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)#把x_image和权值张量进行卷积，加上偏置项，进行relu激活函数
h_pool1 = max_pool_2x2(h_conv1)                        #池化，即简化矩阵
#进行第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])#把类似的层叠起来，然后每个5*5的patch会得到64个特征
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#重复卷积和池化操作
h_pool2 = max_pool_2x2(h_conv2)

#密集卷积层
W_fc1 = weight_variable([16 * 16 * 64, 1024])#缩小图片尺寸，这里长宽都变成之前的1/4，7*7加入一个1024尺寸的全连接层
b_fc1 = bias_variable([1024])               #初始化一个偏置量

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])#将池化层输出的张量转化为一个向量，这里对应上面缩小的图片尺寸
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#表示输出在神经元输出在dropout中保持不变，还可以自动处理神经元输出的scale

#输出层
W_fc2 = weight_variable([1024, 10])#权重
b_fc2 = bias_variable([10])#偏置

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#添加一个softmax层

#训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# reduce_mean  求tensor中平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver() #定义saver
method=Method()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, 'SAVE/model.ckpt')
    for i in range(17000):#17000次训练
        # 每次返回50个训练集的数据
        # batch = mnist.train.next_batch(50)
        batch_data, batch_target=method.next_batch(train_datas,train_targets,64)
        if i % 100 == 0:#每100次迭代输出一次
            train_accuracy = accuracy.eval(feed_dict={
                x : batch_data, y_ : batch_target, keep_prob: 1.0})#在feed_dict中添加keep_dict的比例
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_data, y_: batch_target, keep_prob: 0.5})
    saver.save(sess, 'Model64_64/model.ckpt') #模型储存位置

    # print('work accuracy %g' % accuracy.eval(feed_dict={#最后得出测试最后的准确率
    #     x: train_datas, y_: train_targets, keep_prob: 1.0}))
