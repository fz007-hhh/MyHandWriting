import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class Dataset:
    # 获取路径下的所有图片
    def get_img_file(self,file_name):
        imagelist = []
        for parent, dirnames, filenames in os.walk(file_name):
            for filename in filenames:
                if filename.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    imagelist.append(os.path.join(parent, filename))
            # print(imagelist)
            return imagelist


    # 加载npy文件
    def load_targetnpy(self):
        data = np.load('dataset/self_target.npy')
        for i in range(9):
        # print(len(data))
            print(data[i])
        print()


    # 保存标签为npy文件
    def save_targetnpy(self,width,target):
        arrays = np.full([width,10], 0.0)
        for i in range(width):
            arrays[i][target] = 1.0
        # for i in range(501):
        #     print(arrays[i])
        # mnist_arrays=mnist.train.labels

        final_array=[]
        for i in range(arrays.shape[0]):
            final_array.append(arrays[i])
        # for i in range(mnist_arrays.shape[0]):
        #     final_array.append(mnist_arrays[i])
        return final_array
        # final_array=np.array(final_array)
        # np.save('dataset/self_target.npy', final_array)


    # 将图片制成数据集
    def savedatanpy(self,path):
        # data=np.full([500,784],0.0)
        data=[]
        imglist=self.get_img_file(path)

        for imgpath in imglist:
            img=Image.open(imgpath)
            img = img.convert('L')  # RGB转成灰色
            # plt.imshow(img)
            # plt.show()
            tv = list(img.getdata())  # 返回img的像素序列
            # 转成二分值
            tva=[(255-x)/255.0 for x in tv]
            data.append(np.array(tva))

        # mnist_imgs=mnist.train.images
        # for i in range(mnist_imgs.shape[0]):
        #     data.append(mnist_imgs[i])

        # data= np.array(data)
        return data
        # data=np.reshape(data,(501,28,28))
        # np.save('dataset/self_data.npy', data)


    # 加载图片数据集
    def load_datanpy(self):
        data=np.load('dataset/self_data.npy')
        print()
        # for i in range(len(data)):
        #     print(data[i])


    # 用最邻域插值法将图片扩充为64*64的
    def savenpy64_64(self):
        # 将原本784*1(即28*28)的矩阵扩充为50*50的矩阵
        datas=np.load('dataset/self_data.npy')
        # (55501,28,28)
        datas=np.reshape(datas,(len(datas),28,28))

        bigdata=[]
        for data in datas:
            num=np.full([64,64],0.0)
            # for i in range(28):
            #     for j in range(28):
            #         num[11+i,11+j]=data[i,j]
            # plt.imshow(data,cmap="gray")
            # plt.show()
            num= self.nearest_neighbour(data,[64,64])
            # plt.imshow(num, cmap="gray")
            # plt.show()
            num=np.reshape(num,(4096))
            bigdata.append(num)
        bigdata=np.array(bigdata)
        np.save('dataset/self_data.npy', bigdata)
        print()



    # 最近邻插值
    def nearest_neighbour(self,src, dst_shape):
        # 获取原图维度
        src_height, src_width = src.shape[0], src.shape[1]
        # 计算新图维度
        dst_height, dst_width = dst_shape[0], dst_shape[1]

        dst = np.zeros(shape=(dst_height, dst_width), dtype=np.float)
        for dst_x in range(dst_height):
            for dst_y in range(dst_width):
                # 寻找源图像对应坐标
                src_x = dst_x * (src_width / dst_width)
                src_y = dst_y * (src_width / dst_width)

                # 四舍五入会超出索引，这里采用向下取整，也就是原本1.5->2, 现在是1.5->1
                src_x = int(src_x)
                src_y = int(src_y)

                # 插值
                dst[dst_x, dst_y] = src[src_x, src_y]
        return dst


    def save_newdata(self):
        alldata=[]
        for i in range(10):
            path='D:\\Photo\\dataset02\\'+str(i)
            alldata=alldata+self.savedatanpy(path)

        alldata=np.array(alldata)
        np.save('dataset/self_data.npy',alldata)


    def save_newtarget(self):
        alltarget=[]
        for i in range(10):
            alltarget=alltarget+self.save_targetnpy(400,i)
        alltarget=np.array(alltarget)
        np.save('dataset/self_target.npy', alltarget)


    def combineNpys(self,path1,path2,savepath):
        finaldataset=[]
        dataset1=np.load(path1)
        dataset2=np.load(path2)
        for i in range(dataset1.shape[0]):
            finaldataset.append(dataset1[i])
        for i in range(dataset2.shape[0]):
            finaldataset.append(dataset2[i])
        finaldataset=np.array(finaldataset)
        np.save(savepath,finaldataset)


    def showData(self):
        datas=np.load('self_data_64_64_500.npy')
        targets=np.load('self_target_64_64_500.npy')
        datas=np.reshape(datas,(datas.shape[0],64,64))
        plt.imshow(datas[20011])
        plt.show()

        plt.imshow(datas[55305])
        plt.show()







if __name__=="__main__":
    dataset=Dataset()
    # dataset.savenpy64_64()
    # dataset.savedatanpy('D:\Photo\dataset01')
    # dataset.load_datanpy()
    # dataset.save_targetnpy()
    # dataset.load_targetnpy()
    # dataset.get_img_file('../photos')
    # dataset.save_newdata()
    # dataset.save_newtarget()
    # dataset.combineNpys('self_data.npy','dataset/self_data.npy','self_data.npy')
    # dataset.combineNpys('self_target.npy','dataset/self_target.npy','self_target.npy')
    dataset.showData()
    print()





