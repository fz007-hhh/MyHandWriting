import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from testImg import TestImg


class interface(QWidget):

    def __init__(self):
        super(interface, self).__init__()
        self.testimg = TestImg()
        self.resize(700, 800)
        self.setWindowTitle("且听风吟")

        self.label1 = QLabel(self)
        self.label1.setText("    显示图片")
        self.label1.setFixedSize(500, 400)
        self.label1.move(100, 80)
        self.label1.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color;font-size:10px;font-weight:bold;font-family:宋体;}")

        self.myText = QtWidgets.QTextEdit(self)
        # label文字框的大小
        self.myText.setFixedSize(350, 50)
        self.myText.move(100, 550)
        self.myText.setText("路径为：")
        # 定义字体
        self.myText.setFont(QFont("", 10, QFont.Bold))

        self.myText1 = QtWidgets.QTextEdit(self)
        # label文字框的大小
        self.myText1.setFixedSize(350, 50)
        self.myText1.move(100, 650)
        self.myText1.setText("识别结果为：")
        # 定义字体
        self.myText1.setFont(QFont("", 10, QFont.Bold))

        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(100, 30)
        btn.clicked.connect(self.openimage)

        self.startbtn=QPushButton(self)
        self.startbtn.setText('识别')
        self.startbtn.move(500,30)
        self.startbtn.clicked.connect(self.startRecongnize)


    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*);;*.jpg;;*.png")
        # 填充label
        jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label1.height())
        self.label1.setPixmap(jpg)
        self.myText.setText(imgName)

    def startRecongnize(self):
        # try:
            # PyQt5的QTextEdit中没有text()方法，只有toPlainText()
        str=self.myText.toPlainText()
        self.myText1.setText('识别中...')
        result=self.testimg.startTest(str)
        self.myText1.setText(result)
        # except:
        #     self.myText1.setText("图片路径存在错误!")
        print()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = interface()
    my.show()
    sys.exit(app.exec_())

