from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from work.test01 import SplitPhoto
from work.npy_dataset import Dataset
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def imageprepare(img):
    # opencv转为PIL格式
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.imshow(img)  #显示需要识别的图片
    # plt.show()
    img = img.convert('L')    #RGB转成灰色
    # plt.imshow(img)
    # plt.show()
    tv = list(img.getdata())#返回img的像素序列
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]  # 得到每一个像素点的灰度，最大1为黑，最小0为白
    return tva

def imageprepareByPath(path):
    img = Image.open(path)
    # plt.imshow(img)  #显示需要识别的图片
    # plt.show()
    img = img.convert('L')  # RGB转成灰色
    # plt.imshow(img)
    # plt.show()
    tv = list(img.getdata())  # 返回img的像素序列
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]  # 得到每一个像素点的灰度，最大1为黑，最小0为白
    return tva

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')


def convolute_pool(img):
    tf.reset_default_graph()
    # result即为图像所有像素点的灰度值数组
    result = imageprepareByPath(img)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    print(x_image)

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 密集卷积层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 添加softmax层

    # 模型评估
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()  # 定义saver

    num=''

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'NewModel/model.ckpt')
        prediction = tf.argmax(y_conv, 1)  # 返回对于y_conv预测到的标签值，与真实标签相比较比较是否匹配
        predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)  # 得出测试的结果
        num=str(predint[0])
    return num


# if __name__ == '__main__':
#     img = 'testPic.jpg'
#     n = cutPic.main(img)
#     for i in range(1, 5):
#         img_file = 'pic/pic' + str(i) + '.jpg'
#         #print(img_file)
#         convolute_pool(img_file)

if __name__== '__main__':
    # img='photos/cont01.jpg'
    split=SplitPhoto()
    datas=Dataset()
    datas=datas.get_img_file('D:\Photo\dataset01')
    # imgs = split.opencvdeal('photos/ph-te12.jpg')
    # for imgpath in datas:
    #     imgs=split.opencvdeal(imgpath)
    #     number=''
    #     for img in imgs:
    #         number+=convolute_pool(img)
    #     print("识别的数字结果:",number)
    # imgs = split.opencvdeal(r'D:\Photo\dataset01\489.png')
    for img in datas:
        number = convolute_pool(img)
        print("识别的数字结果:", number)