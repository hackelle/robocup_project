#!/usr/bin/env python2

import logging
import os
import sys
import threading
import argparse

from PyQt5 import QtWidgets, QtGui, QtCore

import image_provider
import vision
import saeros
from logger import GUILogger


class GUI(QtWidgets.QMainWindow, saeros.Ui_MainWindow):
    """Qt GUI Application widget for SAEROS"""

    def __init__(self, args, parent=None):
        super(GUI, self).__init__(parent)
        self._project_path = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )

        self.setupUi(self)
        self.logger = logging.getLogger()
        self.logger.addHandler(GUILogger())
        self.logger.setLevel(logging.DEBUG)
        self.camera.setScaledContents(True)

        self.temps = []
        self.edges = []
        self.edges_old = []

        # actual vision
        self.vision = None
        self.vision_thread = None
        # for keybindings
        self._pause_shortcut = None
        self._next_shortcut = None
        self._prev_shortcut = None
        self.create_vision(args)
        self.create_keybindings()

    def create_vision(self, args):
        if args.img_source == 'rt':
            img_provider = image_provider.RCVisionProvider(args.location)
        elif args.img_source == 'img':
            img_provider = image_provider.StorageVisionProvider(
                os.path.join(args.location)
            )
        elif args.img_source == 'dir':
            img_provider = image_provider.DirectoryVisionProvider(
                os.path.join(args.location)
            )

        self.vision = vision.Vision(
            img_provider,
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
        self.drawing.setPixmap(images['drawing'])

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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Debug the SAERO-service')
    parser.add_argument('img_source', choices=['rt', 'img', 'dir'],
                        help='The type of image source (realtime, single image '
                        'from disk or all images from a directory)')
    parser.add_argument('location', help='The path for the given image source ('
                        'IP or a path to a file/directory)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI(args)
    gui.show()
    app.exec_()


if __name__ == '__main__':
    main()
