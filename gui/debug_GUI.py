#!/usr/bin/env python2

import os
import sys
import threading

from PyQt5 import QtWidgets

import vision
import saeros


class GUI(QtWidgets.QMainWindow, saeros.Ui_MainWindow):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent)
        self._project_path = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )

        self.setupUi(self)
        self.camera.setScaledContents(True)

        self.temps = []
        self.edges = []

        self.create_vision()

    def create_vision(self):
        self.vision = vision.Vision(
            # vision.RCVisionProvider('10.0.7.15'),
            vision.StorageVisionProvider(
                os.path.join(
                    self._project_path,
                    'training-coding/pics/priya_20150326_default_225.png'
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

    def on_vision_updated(self, images):
        self.camera.setPixmap(images['camera'])
        self.show_images(self.temp_inner, images['temp'], self.temps,
                         images['scores'])
        self.show_images(self.edges_inner, images['edges'], self.edges,
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
