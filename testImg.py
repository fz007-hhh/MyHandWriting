from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from work.test01 import SplitPhoto
from work.npy_dataset import Dataset
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TestImg:
    def imageprepare(self,img):
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


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')


    def convolute_pool(self,img):
        tf.reset_default_graph()
        # result即为图像所有像素点的灰度值数组
        result = self.imageprepare(img)

        x = tf.placeholder(tf.float32, [None, 4096])
        y_ = tf.placeholder(tf.float32, [None, 10])
        # 第一层卷积
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])

        x_image = tf.reshape(x, [-1, 64, 64, 1])
        print(x_image)

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # 第二层卷积
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # 密集卷积层
        W_fc1 = self.weight_variable([16 * 16 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 输出
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

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
            saver.restore(sess, 'Model64_64/model.ckpt')
            prediction = tf.argmax(y_conv, 1)  # 返回对于y_conv预测到的标签值，与真实标签相比较比较是否匹配
            predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)  # 得出测试的结果
            num=str(predint[0])
        return num

    def startTest(self,path):
        split = SplitPhoto()
        testImg = TestImg()
        imgs = split.opencvdeal(path)
        number = ''
        for img in imgs:
            number += testImg.convolute_pool(img)
        # print("识别的数字结果:", number)
        return number



def get_img_file(self,file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        # print(imagelist)
        return imagelist
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
    testImg=TestImg()
    # imgs = split.opencvdeal('photos/ph-te12.jpg')
    # imgs = split.opencvdeal('D:\\Photo\\testset-shun\\159.png')
    # number=''
    # for img in imgs:
    #     number+=testImg.convolute_pool(img)
    # print("识别的数字结果:",number)
    path="D:\\Photo\\testset-shun\\"
    numbers=['32' ,'130','1564','562','78','80','69','40','2315','58',
             '73', '1234','1583','529','70','81','70', '181','1325','59',
             '233' ,'11','1574','573','79','18','26','40','85','15',
             '1221','12','134456','256','78','80','69','40','236','5815',
             '123','13','855','55','717','81','69','41','232','671',
             '67','14','154','54','763','82','68','42','231','672',
             '985','15','753','53','735','83','67','43','213','683',
             '67','16','652','52','741','84','66','44','123','769',
             '54','17','1151','51','231','85','65','45','909','721',
             '1515','18','50','15','32','86','64','46','285','533',
             '353','19','49','16','51','87','63','47','235','555',
             '53','10','48','17','23','88','62','48','115','587',
             '420','11','47','18','192','89','61','49','905','5123',
             '21','12','46','19','331','90','60','94','805','5823',
             '22','43','45','20','342','91','59','93','405','822']
    accuracy=0
    for i in range(151,301):
        imgpath=path+str(i)+'.png'
        imgs=split.opencvdeal(imgpath)
        number=''
        for img in imgs:
            number+=testImg.convolute_pool(img)
        if(number==numbers[i-151]):accuracy+=1
        else:print("图片",i,'.png',"识别错误，正确结果应为：",numbers[i-151],"，识别结果为：",number)
    print("测试共150张图片，准确度为：",accuracy/150)
