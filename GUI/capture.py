from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os
import sys
import cv2
from detect import Face_detect
from PIL import ImageQt

class capture_Window(QMainWindow):

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(600, 550)

        self.centralcamera_widget = QWidget()
        self.centralcamera_widget.setObjectName("centralcamera_widget")

        self.verticalLayoutWidget = QWidget(self.centralcamera_widget)
        self.verticalLayoutWidget.setGeometry(QRect(50, 0, 500, 500))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        #find camera
        self.cap = cv2.VideoCapture()
        self.cap.open(0)

        #set timer to update the frame
        self.timer_camera = QTimer()
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera.start(60)

        #face detect
        self.face_detect = Face_detect()

        #show camera 
        self.camera = QLabel(self.verticalLayoutWidget)
        self.camera.setObjectName("camera")
        self.verticalLayout.addWidget(self.camera)

        #capture button
        self.capture = QPushButton(self.verticalLayoutWidget)
        self.capture.setMinimumSize(QSize(50, 50))
        self.verticalLayout.addWidget(self.capture)
        
        self.setCentralWidget(self.centralcamera_widget)
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, 600, 20))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        
        self.retranslateUi()
        QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.capture.setIcon(QIcon('icon/camera.png'))
        self.capture.setIconSize(QSize(50, 50))


    def show_camera(self):
        flag, self.image = self.cap.read()

        img_detections = self.face_detect.detect(self.image)
        self.face_img, self.image = self.face_detect.draw_result(self.image, img_detections)

        show = cv2.resize(self.image, (500, 500))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

        self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.camera.setPixmap(QPixmap.fromImage(self.showImage))


    def capture_pic(self,i):
        self.timer_camera.stop()

        #Face detection
        # img = ImageQt.fromqimage(self.showImage)
        # img_detections = self.face_detect.detect(img)
        # self.face_img, _ = self.face_detect.draw_result(img, img_detections)
        self.face = QImage(ImageQt.ImageQt(self.face_img)).convertToFormat(QImage.Format_ARGB32)

        #save Face
        self.img_path = os.path.join(os.getcwd(),"test image\\opencv_frame_{}.png".format(i))
        self.face.save(self.img_path)

        self.cap.release()
        self.close()

        return self.face,self.img_path
        

        
if __name__ == '__main__':

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()

    window = capture_Window()
    window.setupUi()
    window.show()

    sys.exit(app.exec_())
