import sys
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import *

from GUI.start import start_MainWindow
from GUI.photo import photo_MainWindow
from GUI.slide import slide_MainWindow
from GUI.camera import camera_MainWindow

class Start_Ui(start_MainWindow):
	
	def __init__(self):
		super().__init__()
		self.setupUi()
		self.photo.clicked.connect(self.goPhoto)
		self.camera.clicked.connect(self.goCamera)
	
	def goPhoto(self):
		self.photo = Photo_Ui()
		self.close()
		self.photo.show()

	def goCamera(self):
		self.camera = Camera_Ui()
		self.close()
		self.camera.show()
	


class Photo_Ui(photo_MainWindow):

	def __init__(self):
		super().__init__()
		self.setupUi()
		self.return_page.clicked.connect(self.goStart)
		self.confirm.clicked.connect(self.goSlide)

	def goStart(self):
		self.start = Start_Ui()
		self.close()
		self.start.show()

	def goSlide(self):
		if self.imagePath == None:
			self.alert("No image")
		else:
			self.slide = Slide_Ui()
			self.slide.Image(self.imagePath)
			self.close()
			self.slide.show()

	def alert(self, s):
		err = QErrorMessage(self)
		err.showMessage(s)


class Camera_Ui(camera_MainWindow):

	def __init__(self):
		super().__init__()
		self.setupUi()
		self.return_page.clicked.connect(self.goStart)
		self.confirm.clicked.connect(self.goSlide)

	def goStart(self):
		self.start = Start_Ui()
		self.close()
		self.start.show()

	def goSlide(self):
		if self.imagePath == None:
			self.alert("No image")
		else:
			self.slide = Slide_Ui()
			self.slide.Image(self.imagePath)
			self.close()
			self.slide.show()

	def alert(self, s):
		err = QErrorMessage(self)
		err.showMessage(s)


class Slide_Ui(slide_MainWindow):

	def __init__(self):
		super().__init__()
		self.setupUi()
		self.return_page.clicked.connect(self.goStart)

	def goStart(self):
		self.start = Start_Ui()
		self.close()
		self.start.show()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	MainWindow = QMainWindow()

	Control = Start_Ui()
	Control.show()

	sys.exit(app.exec_())