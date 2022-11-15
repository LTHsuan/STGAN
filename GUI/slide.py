from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL.ImageQt import ImageQt

import sys
import argparse
from agents import STGANAgent
from utils.config import *
from PIL import Image



class slide_MainWindow(QMainWindow):
    def setupUi(self):

        self.setObjectName("MainWindow")
        self.resize(800, 600)

        self.STGAN_init()

        self.centralwidget = QWidget()
        self.centralwidget.setObjectName("centralwidget")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setGeometry(QRect(180, 340, 521, 191))
        self.splitter.setMouseTracking(False)
        self.splitter.setAcceptDrops(False)
        self.splitter.setOrientation(Qt.Vertical)

        #Return to start page
        self.return_page = QPushButton(self.centralwidget)
        self.return_page.setGeometry(QRect(10, 10, 50, 50))
        self.return_page.setObjectName("return_page")

        #Return to start page
        self.reset = QPushButton(self.centralwidget)
        self.reset.setGeometry(QRect(60, 10, 50, 50))
        self.reset.setObjectName("reset_att")

        # Attribute Slider
        self.splitter.setObjectName("splitter")

        self.att1_slider = QSlider(self.splitter)
        self.att1_slider.setPageStep(1)
        self.att1_slider.setOrientation(Qt.Horizontal)
        self.att1_slider.setInvertedAppearance(False)
        self.att1_slider.setObjectName("att1_slider")
        

        self.att2_slider = QSlider(self.splitter)
        self.att2_slider.setPageStep(1)
        self.att2_slider.setOrientation(Qt.Horizontal)
        self.att2_slider.setInvertedAppearance(False)
        self.att2_slider.setObjectName("att2_slider")
        

        self.att3_slider = QSlider(self.splitter)
        self.att3_slider.setPageStep(1)
        self.att3_slider.setOrientation(Qt.Horizontal)
        self.att3_slider.setInvertedAppearance(False)
        self.att3_slider.setObjectName("att3_slider")
        

        self.att4_slider = QSlider(self.splitter)
        self.att4_slider.setPageStep(1)
        self.att4_slider.setOrientation(Qt.Horizontal)
        self.att4_slider.setInvertedAppearance(False)
        self.att4_slider.setObjectName("att4_slider")

        self.att5_slider = QSlider(self.splitter)
        self.att5_slider.setPageStep(1)
        self.att5_slider.setOrientation(Qt.Horizontal)
        self.att5_slider.setInvertedAppearance(False)
        self.att5_slider.setObjectName("att5_slider")


        # Attribute Label
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QRect(90, 340, 81, 191))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.att1 = QLabel(self.layoutWidget)
        self.att1.setObjectName("att1")
        self.att1.setFont(QFont('Arial', 10))
        self.verticalLayout.addWidget(self.att1)

        self.att2 = QLabel(self.layoutWidget)
        self.att2.setObjectName("att2")
        self.att2.setFont(QFont('Arial', 10))
        self.verticalLayout.addWidget(self.att2)

        self.att3 = QLabel(self.layoutWidget)
        self.att3.setObjectName("att3")
        self.att3.setFont(QFont('Arial', 10))
        self.verticalLayout.addWidget(self.att3)

        self.att4 = QLabel(self.layoutWidget)
        self.att4.setObjectName("att4")
        self.att4.setFont(QFont('Arial', 10))
        self.verticalLayout.addWidget(self.att4)

        self.att5 = QLabel(self.layoutWidget)
        self.att5.setObjectName("att5")
        self.att5.setFont(QFont('Arial', 10))
        self.verticalLayout.addWidget(self.att5)

        #Image Show
        self.layoutWidget1 = QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QRect(100, 0, 640, 350))
        self.layoutWidget1.setObjectName("layoutWidget1")

        self.horizontalLayout = QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(50, 50, 50, 50)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.original_img = QLabel(self.layoutWidget1)
        self.original_img.setObjectName("original_img")
        self.horizontalLayout.addWidget(self.original_img)

        self.att_img = QLabel(self.layoutWidget1)
        self.att_img.setObjectName("att_img")
        self.horizontalLayout.addWidget(self.att_img)

        # Menubar statusbar
        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi(self)
        QMetaObject.connectSlotsByName(self)

        self.slider_change()
        self.reset.clicked.connect(self.setvalue)


    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.return_page.setIcon(QIcon('icon\\home.png')) 
        self.return_page.setIconSize(QSize(50,50))
        self.reset.setIcon(QIcon('icon\\return.png')) 
        self.reset.setIconSize(QSize(50,50))


        self.att1.setText(_translate("MainWindow", "Brown_Hair"))
        self.att2.setText(_translate("MainWindow", "Eyeglasses"))
        self.att3.setText(_translate("MainWindow", "Goatee"))
        self.att4.setText(_translate("MainWindow", "Pale_Skin"))
        self.att5.setText(_translate("MainWindow", "Rosy_Cheeks"))

        

    def slider_change(self):
        # Slider change 
        self.att = [self.att1_slider.value(),self.att2_slider.value(),self.att3_slider.value(),self.att4_slider.value(),self.att5_slider.value()]

        self.att1_slider.sliderReleased.connect(lambda: self.valuechange())
        self.att2_slider.sliderReleased.connect(lambda: self.valuechange())
        self.att3_slider.sliderReleased.connect(lambda: self.valuechange())
        self.att4_slider.sliderReleased.connect(lambda: self.valuechange())
        self.att5_slider.sliderReleased.connect(lambda: self.valuechange())

    def STGAN_init(self):
        #create the STGAN model
        arg_parser = argparse.ArgumentParser(description='Pytorch Classfication Train')
        arg_parser.add_argument(
            '--config',
            default="./configs/train_stgan.yaml",
            help='The path of configuration file in yaml format')
        args = arg_parser.parse_args()
        config = process_config(args.config)
        self.agent = STGANAgent(config)

    def Image(self,imagePath):
        self.imgPath = imagePath

        self.org_att, self.org_img = self.agent.Creat_attrs_org(self.imgPath)
        print("orginal att = ",self.org_att)
        #self.org_img.show()
        

        self.att1_slider.setMinimum(-abs(self.org_att[0]*10000))
        self.att1_slider.setMaximum(abs(self.org_att[0]*10000))
        print("att1",-abs(self.org_att[0]*10000),"~",abs(self.org_att[0]*10000))

        self.att2_slider.setMinimum(-abs(self.org_att[1]*10000))
        self.att2_slider.setMaximum(abs(self.org_att[1]*10000))
        print("att2",-abs(self.org_att[1]*10000),"~",abs(self.org_att[1]*10000))

        self.att3_slider.setMinimum(-abs(self.org_att[2]*10000))
        self.att3_slider.setMaximum(abs(self.org_att[2]*10000))
        print("att3",-abs(self.org_att[2]*10000),"~",abs(self.org_att[2]*10000))

        self.att4_slider.setMinimum(-abs(self.org_att[3]*10000))
        self.att4_slider.setMaximum(abs(self.org_att[3]*10000))
        print("att4",-abs(self.org_att[3]*10000),"~",abs(self.org_att[3]*10000))

        self.att5_slider.setMinimum(-abs(self.org_att[4]*10000))
        self.att5_slider.setMaximum(abs(self.org_att[4]*10000))
        print("att5",-abs(self.org_att[4]*10000),"~",abs(self.org_att[4]*10000))

        self.setvalue()


    def setvalue(self):
        Img = QImage(ImageQt(self.org_img)).convertToFormat(QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(Img).scaled(200, 200)
        self.original_img.setPixmap(pixmap)

        self.att1_slider.setProperty("value", self.org_att[0]*10000)
        self.att2_slider.setProperty("value", self.org_att[1]*10000)
        self.att3_slider.setProperty("value", self.org_att[2]*10000)
        self.att4_slider.setProperty("value", self.org_att[3]*10000)
        self.att5_slider.setProperty("value", self.org_att[4]*10000)
        self.valuechange()


    def valuechange(self):
        self.att = self.org_att.copy()
        self.att = [self.att1_slider.value()/10000,self.att2_slider.value()/10000,self.att3_slider.value()/10000,self.att4_slider.value()/10000,self.att5_slider.value()/10000]
        print("sliding att = ",self.att)
        self.GenerateImg()


    def GenerateImg(self): 
        self.att_image = self.agent.Generate_by_face(self.org_img,self.org_att,self.att)
        Img = QImage(ImageQt(self.att_image)).convertToFormat(QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(Img).scaled(200, 200)
        self.att_img.setPixmap(pixmap)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = slide_MainWindow()
    ui.setupUi()
    ui.show()

    sys.exit(app.exec_())
