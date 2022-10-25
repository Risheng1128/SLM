from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from recoat_trainer import *
import cv2 as cv
import time
import imghdr
import sys
import os

def open_file():
    path, _ = QFileDialog.getOpenFileName(None)
    if not path:
        print("No image! Please select a image")
        return None

    filetype = imghdr.what(path)
    if (filetype != "jpeg" and filetype != "png" and filetype != "bmp"):
        print("File type error!")
        return None
    return path

def open_dir():
    path = QFileDialog.getExistingDirectory(None)
    if not path:
        print("Please select a folder")
        return None
    return path

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # initial model parameter
        self.output_path = 'Result/Detect/'
        self.model_path = './Model/recoat.pth'
        self.model = Detector(dst_path=self.output_path)

        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1770, 900)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # initial origin image window
        self.origin_window = QtWidgets.QGraphicsView(self.centralwidget)
        self.origin_window.setGeometry(QtCore.QRect(50, 60, 700, 700))
        self.origin_window_scene = QtWidgets.QGraphicsScene()
        self.origin_window_scene.setSceneRect(50, 60, 700, 700)

        # initial image detection window
        self.detect_window = QtWidgets.QGraphicsView(self.centralwidget)
        self.detect_window.setGeometry(QtCore.QRect(800, 60, 700, 700))
        self.detect_window_scene = QtWidgets.QGraphicsScene()
        self.detect_window_scene.setSceneRect(800, 60, 700, 700)

        # display file system window
        self.file_system = QtWidgets.QFileSystemModel(self.centralwidget)
        self.file_system.setNameFilters(["*.png", "*.jpg", "*.bmp"])
        self.file_system.setNameFilterDisables(False)
        self.list = QtWidgets.QListView(self.centralwidget)
        self.list.setModel(self.file_system)
        self.list.setGeometry(QtCore.QRect(1525, 280, 220, 480))
        self.list.clicked.connect(self.click_file_system)

        # initial label
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(765, 10, 250, 30))
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)

        self.origin_label = QtWidgets.QLabel(self.centralwidget)
        self.origin_label.setGeometry(QtCore.QRect(325, 760, 250, 30))
        self.origin_label.setFont(font)
        self.origin_label.setLayoutDirection(QtCore.Qt.LeftToRight)

        self.detection_label = QtWidgets.QLabel(self.centralwidget)
        self.detection_label.setGeometry(QtCore.QRect(1100, 760, 250, 30))
        self.detection_label.setFont(font)
        self.detection_label.setLayoutDirection(QtCore.Qt.LeftToRight)

        self.defects_label = QtWidgets.QLabel(self.centralwidget)
        self.defects_label.setGeometry(QtCore.QRect(1525, 60, 250, 30*7))
        self.defects_label.setFont(font)
        self.defects_label.setLayoutDirection(QtCore.Qt.LeftToRight)

        self.time_label = QtWidgets.QLabel(self.centralwidget)
        self.time_label.setGeometry(QtCore.QRect(1675, 233, 250, 30))
        self.time_label.setFont(font)
        self.time_label.setLayoutDirection(QtCore.Qt.LeftToRight)

        # initial load image button
        self.load_img_button = QtWidgets.QPushButton(self.centralwidget)
        self.load_img_button.setGeometry(QtCore.QRect(540, 825, 220, 40))
        self.load_img_button.setFont(font)
        self.load_img_button.clicked.connect(self.click_load_img_button)

        # initial image detection button
        self.detect_img_button = QtWidgets.QPushButton(self.centralwidget)
        self.detect_img_button.setGeometry(QtCore.QRect(765, 825, 220, 40))
        self.detect_img_button.setFont(font)
        self.detect_img_button.clicked.connect(self.click_detect_img_button)

        # initial the repository image detection button
        self.load_dir_button = QtWidgets.QPushButton(self.centralwidget)
        self.load_dir_button.setGeometry(QtCore.QRect(990, 825, 220, 40))
        self.load_dir_button.setFont(font)
        self.load_dir_button.clicked.connect(self.click_load_dir_button)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText("鋪粉後瑕疵檢測系統")
        self.origin_label.setText("原始圖片")
        self.detection_label.setText("辨識後圖片")
        self.defects_label.setText("瑕疵分類:\n0: 鋪粉不均勻\n1: 鋪粉未覆蓋\n2: 刮刀刮痕\n辨識時間(s):")
        self.load_img_button.setText("讀取圖片")
        self.detect_img_button.setText("讀取圖片並辨識")
        self.load_dir_button.setText("選擇目錄")

    def click_load_img_button(self):
        filepath = open_file()
        if not filepath:
            return
        self.display_graphics_view(filepath, self.origin_window, self.origin_window_scene, 50)

    def click_detect_img_button(self):
        filepath = open_file()
        if not filepath:
            return
        QtWidgets.QApplication.processEvents()
        self.display_origin_and_detect_img(filepath)

    def click_load_dir_button(self):
        folderpath = open_dir()
        if not folderpath:
            return
        self.list.setRootIndex(self.file_system.setRootPath(folderpath))

    def click_file_system(self, filepath):
        path = self.file_system.fileInfo(filepath).absoluteFilePath()
        if os.path.isdir(path):
            self.list.setRootIndex(self.file_system.setRootPath(path))
            return
        self.display_origin_and_detect_img(path)

    def display_origin_and_detect_img(self, filepath):
        print("dtetect file", filepath)
        self.display_graphics_view(filepath, self.origin_window, self.origin_window_scene, 50)
        start = time.time()
        self.model.Save_Prediction(filepath, self.output_path, self.model_path)
        end = time.time()
        print("detection time: ", end - start, "s")
        self.time_label.setText(str(round(end - start, 3)))
        self.display_graphics_view(self.output_path + filepath.split('/')[-1], self.detect_window, self.detect_window_scene, 800)

    def display_graphics_view(self, filename, view, scene, x_offset):
        img = QPixmap(filename)
        img = img.scaled(700, 700)
        item = QtWidgets.QGraphicsPixmapItem(img)
        item.setOffset(x_offset, 60)
        scene.addItem(item)
        view.setScene(scene)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
