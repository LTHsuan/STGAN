from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import sys 


class start_MainWindow(QMainWindow):

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(800, 600)

        self.centralwidget = QWidget()
        self.centralwidget.setObjectName("centralwidget")

        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QRect(140, 320, 491, 191))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        #title
        self.title = QLabel(self.centralwidget)
        self.title.setGeometry(QRect(150,50,540,200))
        self.title.setObjectName("title")
        self.title.setFont(QFont('Arial', 60, QFont.Bold))

        #camera icon button
        self.camera = QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera.sizePolicy().hasHeightForWidth())
        self.camera.setSizePolicy(sizePolicy)
        self.camera.setObjectName("camera")
        self.horizontalLayout.addWidget(self.camera)

        #photo icon button
        self.photo = QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.photo.sizePolicy().hasHeightForWidth())
        self.photo.setSizePolicy(sizePolicy)
        self.photo.setObjectName("photo")
        self.horizontalLayout.addWidget(self.photo)

        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, 800, 21))
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

        self.title.setText(' S  T  G  A  N ')
        self.camera.setIcon(QIcon('icon\\camera.png')) 
        self.camera.setIconSize(QSize(100,100))
        self.photo.setIcon(QIcon('icon\\photo.png'))
        self.photo.setIconSize(QSize(100,100))



if __name__ == '__main__': 

    app = QApplication(sys.argv)

    ui = start_MainWindow()
    ui.setupUi()
    ui.show()

    sys.exit(app.exec_())
