from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2


class SplitPhoto:

    def cv2convert(self,path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img,(150,150))
        # 自适应阙值图形算法,blocksize必须为奇数，blocksize越大，越趋近二值化图像
        newImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 37, 5)

        # 定义结构元素，膨胀结构为原像素点为中心的3*3矩阵
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # 图像膨胀，保证数字像素点集准确粘连，尤其是数字5，如果不粘连会形成两个轮廓
        newImg = cv2.dilate(newImg, kernel)

        plt.imshow(newImg, cmap="gray")
        plt.show()

        cv2.imwrite('D:\\PyCharm_Work\\Python_Basic\\TensorFlow_Study\\MyHandWriting\\photos\\2314.jpg',newImg)



    def blackCastToWhite(self, mutex):
        m1 = np.full([3], 0, np.uint8)
        m2 = np.full([3], 255, np.uint8)
        for i in mutex:
            # 这个分区段为白色，改为黑色
            if (i <= 255 and i > 130):
                return m1
            # 剩余分区段默认为黑色，改成白色
            else:
                return m2


    def black_white_con(self, mutex):
        m1 = np.full([3], 0, np.uint8)
        m2 = np.full([3], 255, np.uint8)
        for i in mutex:
            if (i < 130):
                return m1
            else:
                return m2



    def opencvdeal(self,path):
        # 为了更高的准确率，使用二值图像。
        self.cv2convert(path)
        img=cv2.imread('D:\PyCharm_Work\\Python_Basic\\TensorFlow_Study\\MyHandWriting\\photos\\2314.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        '''
        findContours有两个返回值，
          1.contours是一个ndarray的list，存储轮廓本身，
        每一个轮廓都是一个ndarray，即轮廓上点的集合
          2.hierarchy存储轮廓的4个属性，是一个二维数组，
        每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]
        '''
        # 使用RETR_TREE会将所有轮廓检查出来，比如数字8会查出最外侧的’8‘、'8’的上部分圆圈、‘8’的下部分圆圈
        # 出于应用考虑，选择RETR_EXTERNAL，只检查最外层就可以了
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 查看提取的轮廓数量
        print('降噪前的轮廓数量:', len(contours))
        # 去除小面积噪点
        cv_contours=[]
        for contour in contours:
            print(cv2.contourArea(contour))
            if(cv2.contourArea(contour)>30):
                cv_contours.append(contour)
            else:
                continue

        print('降噪后的轮廓数量:', len(cv_contours))

        photos=[]
        for i in range(len(cv_contours)):
            new_img = 0
            # 利用轮廓近似多边形提取数字信息
            box = cv2.approxPolyDP(cv_contours[i], 0.0001, True)
            # 获取数字部分的矩形区域
            x, y, w, h = cv2.boundingRect(cv_contours[i])

            box = box.reshape(-1, 1, 2)
            # 定义一张和所选图片大小相同的图片，矩阵元素都为0，即全黑
            ploygon = np.zeros(img.shape, np.uint8)
            # 用白色勾勒轮廓边界线
            ploygon = cv2.polylines(ploygon, [box], True, (255, 255, 255))
            # 用白色填充轮廓
            final_ploy = cv2.fillPoly(ploygon, [box], (255, 255, 255))
            # 用轮廓和原图片进行位运算，即在原来的图片上画出轮廓
            ploy_img = cv2.bitwise_and(final_ploy, img)

            ploy_img = ploy_img[y:y + h, x:x + w]

            # plt.imshow(ploy_img)
            # plt.show()

            # 图片居中
            h = ploy_img.shape[0]
            w = ploy_img.shape[1]
            if (h >= w):
                new_h = int(float(1.4 * h))
                new_w = new_h
                sub_w = int((new_w - w) / 2)
                sub_h = int(h / 5)

                new_img = np.full([new_h, new_w, 3], 255, np.uint8)
                for i in range(sub_h, new_h - sub_h - 1):
                    for j in range(sub_w, new_w - sub_w - 1):
                        mutex = ploy_img[i - sub_h, j - sub_w]
                        mutex = self.blackCastToWhite(mutex)
                        new_img[i, j] = mutex

                # plt.imshow(new_img)
                # plt.show()

                # 压缩成28*28，否则模型不能读取
                # 使用PIL高质量压缩，opencv压缩的质量较差
                # new_img = cv2.resize(new_img, (28, 28))
                pil_image=Image.fromarray(cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB))
                pil_image=pil_image.resize((28,28),Image.ANTIALIAS)

                plt.imshow(pil_image)
                plt.show()

                # 由PIL转回opencv格式，进行赋值运算
                new_img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

                # 压缩过程中有部分像素颜色有偏差，简单恢复一下
                for i in range(28):
                    for j in range(28):
                        mutex = new_img[i, j]
                        mutex = self.black_white_con(mutex)
                        new_img[i, j] = mutex
                plt.imshow(new_img)
                plt.show()
                photos.append(new_img)

        return photos

