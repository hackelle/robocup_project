#!/usr/bin/env python2

import logging
import os
import sys
import threading

from PyQt5 import QtWidgets, QtGui, QtCore

import vision
import saeros
from logger import GUILogger


class GUI(QtWidgets.QMainWindow, saeros.Ui_MainWindow):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent)
        self._project_path = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )

        self.setupUi(self)
        self.logger = logging.getLogger()
        self.logger.addHandler(GUILogger())  # self.log_view))
        self.logger.setLevel(logging.DEBUG)
        self.camera.setScaledContents(True)

        self.temps = []
        self.edges = []
        self.edges_old = []

        self.create_vision()
        self.create_keybindings()

    def create_vision(self):
        self.vision = vision.Vision(
            # vision.RCVisionProvider('10.0.7.14'),
            # vision.StorageVisionProvider(
            #     os.path.join(
            #         self._project_path,
            #         # 'training-coding/pics/priya_20150326_default_225.png'
            #         'reference-images/full',
            #         'converted-two-robots-2.png'
            #     )
            # ),
            vision.DirectoryVisionProvider(
                os.path.join(
                    self._project_path,
                    'reference-images',
                    'two-robots'
                )
            ),
            os.path.join(
                self._project_path,
                'training-coding/models/roboheads-ssd_mobilenet_v1',
                'frozen_inference_graph.pb'
            )
        )
        self.vision.updated.connect(self.on_vision_updated)

        self.vision_thread = threading.Thread(target=self.vision.run)
        self.vision_thread.start()

    def create_keybindings(self):
        self._pause_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Space),
            self
        )
        self._pause_shortcut.activated.connect(self.vision.pause)
        self._next_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Right),
            self
        )
        self._next_shortcut.activated.connect(self.vision.next)
        self._prev_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Left),
            self
        )
        self._prev_shortcut.activated.connect(self.vision.prev)

    def on_vision_updated(self, images):
        self.camera.setPixmap(images['camera'])
        self.show_images(self.temp_inner, images['temp'], self.temps,
                         images['scores'])
        self.show_images(self.edges_inner, images['edges'], self.edges,
                         images['scores'])
        self.show_images(self.edges_inner_old, images['edges'], self.edges_old,
                         images['scores'])
        # self.edges.setPixmap(images['edges'])
        # self.temp.setPixmap(images['temp'])

    def show_images(self, tabs, images, labels, scores):
        tabs.clear()
        while len(labels) > 0:
            label = labels.pop()
            label.close()
            label.deleteLater()

        for i, img in enumerate(images):
            label = QtWidgets.QLabel(tabs)
            label.setPixmap(img)
            label.setScaledContents(True)
            tabs.addTab(label, str(scores[i]))
            label.show()
            labels.append(label)

    def closeEvent(self, _):
        self.vision.stop()
        self.vision_thread.join()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.show()
    app.exec_()
