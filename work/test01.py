from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2


class SplitPhoto:

    # 降噪，水泛染+面积选择降噪
    def deNoise(self,temImg):
        temImg=self.FloodNoise(temImg)
        temImg=self.AreaNoise(temImg)
        return temImg

    # 水泛染降噪法,去除大噪点
    def FloodNoise(self, temImg):
        # 遍历所有点，此时每团连接段的灰度值都不一样，计算每种灰度值的连接点数量即得轮廓大小
        # 这里绝不能用np.uint8，否则点数超过255会自动置0
        color_count=np.full([256],0,np.int)
        for i in range(temImg.shape[0]):
            for j in range(temImg.shape[1]):
                if(temImg[i,j]!=0):
                    color_count[temImg[i, j]] += 1
        # plt.imshow(temImg, cmap="gray")
        # plt.show()
        # 正式降噪，数量小于20就置为黑色
        for i in range(temImg.shape[0]):
            for j in range(temImg.shape[1]):
                if(color_count[temImg[i,j]]<=20):
                    temImg[i,j]=0
        # 降噪已清除，将连接点颜色置为白色，图片中残留的是较小的噪点
        for i in range(temImg.shape[0]):
            for j in range(temImg.shape[1]):
                if(temImg[i,j]<255):
                    temImg[i,j]=0
        # plt.imshow(temImg, cmap="gray")
        # plt.show()
        return temImg


    # 轮廓面积降噪法，清理小面积噪点
    def AreaNoise(self, temImg):
        contours = self.getcontours(temImg)
        for i in range(len(contours)):
            # print(cv2.contourArea(contour))
            if (cv2.contourArea(contours[i]) < 25):
                cv2.drawContours(temImg,contours,i,(0,0,0),6)
            else:
                continue
        # plt.imshow(temImg, cmap="gray")
        # plt.show()
        return temImg

    # 获取图片的全部轮廓
    def getcontours(self, temImg):
        cv2.imwrite('D:\\PyCharm_Work\\Python_Basic\\TensorFlow_Study\\MyHandWriting\\photos\\2314.jpg', temImg)
        temImg = cv2.imread('D:\PyCharm_Work\\Python_Basic\\TensorFlow_Study\\MyHandWriting\\photos\\2314.jpg')
        gray = cv2.cvtColor(temImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # findContours有两个返回值，
        #   1.contours是一个ndarray的list，存储轮廓本身，
        # 每一个轮廓都是一个ndarray，即轮廓上点的集合
        #   2.hierarchy存储轮廓的4个属性，是一个二维数组，
        # 每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 使用RETR_TREE会将所有轮廓检查出来，比如数字8会查出最外侧的’8‘、'8’的上部分圆圈、‘8’的下部分圆圈
        # 出于应用考虑，选择RETR_EXTERNAL，只检查最外层就可以了
        return contours


    # 给轮廓排序，排序规则为从左到右
    def sortContour(self, contours):
        # contours是一个元组，元组不能互相赋值，所以用一个list暂时储存contours，然后再排序
        contourlist=[]
        for i in range(len(contours)):
            contourlist.append(contours[i])
        # 冒泡排序
        for i in range(len(contourlist)):
            for j in range(i+1, len(contourlist)):
                x1, y1, w1, h1 = cv2.boundingRect(contourlist[i])
                x2, y2, w2, h2 = cv2.boundingRect(contourlist[j])
                if(x1>x2):
                    tem=contourlist[i]
                    contourlist[i]=contourlist[j]
                    contourlist[j]=tem
        return contourlist


    # 黑白色转换
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


    # 黑白色转换
    def black_white_con(self, mutex):
        m1 = np.full([3], 0, np.uint8)
        m2 = np.full([3], 255, np.uint8)
        for i in mutex:
            if (i < 140):
                return m1
            else:
                return m2


    # 清除模糊像素
    def killVague(self, gray, i, j):
        if(int(gray[i,j])<200):
            return 0
        else:
            return 255


    # 对图片进行灰度、二值、膨胀、降噪处理
    def cv2convert(self,path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img,(150,150))
        # 自适应阙值图形算法,blocksize必须为奇数，blocksize越大，越趋近二值化图像
        newImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 37, 5)
        # plt.imshow(newImg, cmap="gray")
        # plt.show()

        # 获取图片的宽高
        h=img.shape[0]
        w=img.shape[1]

        if(h>200 or w>200):
            newImg= cv2.resize(newImg, (200, int(200*(h/w))))
        # plt.imshow(newImg,cmap="gray")
        # plt.show()

        bestImg=0
        avail_contournum=0
        contournum=10000
        # 定义结构元素，膨胀结构为原像素点为中心的n*n矩阵
        # 由于不同图片的膨胀合适范围不同，对图片进行膨胀后再降噪，计算出轮廓数的最小值（即数字位数）
        for i in range(2,4):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (i, i))
            # 图像膨胀，保证数字像素点集准确粘连，尤其是数字5，如果不粘连会形成两个轮廓
            temImg=cv2.dilate(newImg, kernel)
            # plt.imshow(temImg, cmap="gray")
            # plt.show()
            temImg=self.deNoise(temImg)
            # plt.imshow(temImg, cmap="gray")
            # plt.show()
            # 计算降噪之后的轮廓数量
            cv_contours=self.getcontours(temImg)
            avail_contour=[]
            for contour in cv_contours:
                if(cv2.contourArea(contour)>30):avail_contour.append(avail_contour)
            if(len(avail_contour) > avail_contournum and len(cv_contours)<=contournum):
                avail_contournum=len(avail_contour)
                contournum=len(cv_contours)
                bestImg=temImg

        # plt.imshow(bestImg, cmap="gray")
        # plt.show()
        return bestImg


    # 分离图片中的数字，并将每个数字保存为28*28的格式
    def opencvdeal(self,path):
        # 为了更高的准确率，使用二值图像。
        dealimg = self.cv2convert(path)
        cv2.imwrite('D:\PyCharm_Work\\Python_Basic\\TensorFlow_Study\\MyHandWriting\\photos\\2314.jpg',dealimg)
        img = cv2.imread('D:\PyCharm_Work\\Python_Basic\\TensorFlow_Study\\MyHandWriting\\photos\\2314.jpg')
        cv_contours=self.getcontours(dealimg)
        cv_contours=self.sortContour(cv_contours)
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
            new_h = 0
            new_w = 0
            sub_w = 0
            sub_h = 0
            if (h >= w):
                new_h = int(float(1.5 * h))
                new_w = new_h
                sub_w = int((new_w - w) / 2)
                sub_h = int(h / 4)
            else:
                new_w = int(float(1.5 * w))
                new_h = new_w
                sub_w = int(w/4)
                sub_h = int((new_h-h)/2)

            new_img = np.full([new_h, new_w, 3], 255, np.uint8)
            for i in range(sub_h, new_h - sub_h - 1):
                for j in range(sub_w, new_w - sub_w - 1):
                    mutex = ploy_img[i - sub_h, j - sub_w]
                    mutex = self.blackCastToWhite(mutex)
                    new_img[i, j] = mutex

            # plt.imshow(new_img)
            # plt.show()

            # 压缩成64*64，否则模型不能读取
            # 使用PIL高质量压缩，opencv压缩的质量较差
            # new_img = cv2.resize(new_img, (28, 28))
            pil_image=Image.fromarray(cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB))
            pil_image=pil_image.resize((64,64),Image.ANTIALIAS)
            # plt.imshow(pil_image)
            # plt.show()

            # 由PIL转回opencv格式，处理像素色彩偏差
            pil_image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_BGR2GRAY)
            new_img = pil_image
            for i in range(64):
                for j in range(64):
                    new_img[i,j]=self.killVague(pil_image,i,j)

            plt.imshow(new_img,cmap='gray')
            plt.show()
            photos.append(new_img)

        return photos

if __name__== '__main__':
    split=SplitPhoto()
    img = cv2.imread('../photos/ph-te12.jpg', cv2.IMREAD_GRAYSCALE)
    newImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 37, 5)
    h = img.shape[0]
    w = img.shape[1]
    if (h > 200 or w > 200):
        newImg = cv2.resize(newImg, (200, int(200 * (h / w))))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 图像膨胀，保证数字像素点集准确粘连，尤其是数字5，如果不粘连会形成两个轮廓
    # newImg = cv2.dilate(newImg, kernel)
    newImg=split.FloodNoise(newImg)
    split.AreaNoise(newImg)