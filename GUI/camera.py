from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image
from GUI.capture import capture_Window


class camera_MainWindow(QMainWindow):
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(800, 600)

        self.centralwidget = QWidget()
        self.centralwidget.setObjectName("centralwidget")

        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QRect(80, 450, 681, 80))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        #Return to start page
        self.return_page = QPushButton(self.centralwidget)
        self.return_page.setGeometry(QRect(10, 10, 50, 50))
        self.return_page.setObjectName("return_page")

        
        #Image load button
        self.camera_window = None
        self.camera = QPushButton(self.horizontalLayoutWidget)
        self.camera.setObjectName("camera")
        self.horizontalLayout.addWidget(self.camera)
        self.camera.clicked.connect(self.open_camera) 
        self.capture_times = 0

        #check Image button
        self.confirm = QPushButton(self.horizontalLayoutWidget)
        self.confirm.setObjectName("confirm")
        self.horizontalLayout.addWidget(self.confirm)

        #show Image
        self.imagePath = None
        self.graphicsView = QLabel(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")


        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, 840, 21))
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

        self.return_page.setIcon(QIcon('icon\\home.png')) 
        self.return_page.setIconSize(QSize(50,50))

        self.camera.setText(_translate("MainWindow", "open camera"))
        self.confirm.setText(_translate("MainWindow", "confirm"))   

    def open_camera(self):
        self.camera_window = capture_Window()
        self.camera_window.setupUi() 
        self.camera_window.show()
        self.camera_window.capture.clicked.connect(self.capture_img)

    def capture_img(self):
        self.img, self.imagePath = self.camera_window.capture_pic(self.capture_times)
        self.capture_times +=1

        pixmap = QPixmap.fromImage(self.img)
        self.graphicsView.setGeometry(QRect(400-pixmap.size().width()/2, 50, 700, 400))
        self.graphicsView.setPixmap(pixmap)

    def alert(self, s):
        err = QErrorMessage(self)
        err.showMessage(s)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = camera_MainWindow()
    ui.setupUi()
    ui.show()

    sys.exit(app.exec_())

