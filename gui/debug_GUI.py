#!/usr/bin/env python2

import sys
import threading

from PyQt5 import QtWidgets, QtGui

import vision
import saeros


class GUI(QtWidgets.QMainWindow, saeros.Ui_MainWindow):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent)
        self.setupUi(self)

        self.create_vision()

    def create_vision(self):
        self.vision = vision.Vision(
            vision.StorageVisionProvider(
                '/home/jasper/dev/robocup_project/reference-images'
                '/full/converted-1m-front-straight.png'
            )
        )
        self.vision.updated.connect(self.on_vision_updated)

        self.vision_thread = threading.Thread(target=self.vision.run)
        self.vision_thread.start()

    def on_vision_updated(self, images):
        self.camera.setPixmap(images['camera'])
        self.edges.setPixmap(images['temp'])

    def closeEvent(self, _):
        self.vision.stop()
        self.vision_thread.join()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.show()
    app.exec_()
