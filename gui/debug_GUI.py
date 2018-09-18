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

        self.create_vision()

    def create_vision(self):
        self.vision = vision.Vision(
            vision.StorageVisionProvider(
                os.path.join(
                    self._project_path,
                    'reference-images/full/converted-1m-front-left-full.png'
                )
            ),
            os.path.join(
                self._project_path,
                'training-coding/models/roboheads-ssd_mobilenet_v1/frozen_inference_graph.pb'
            )
        )
        self.vision.updated.connect(self.on_vision_updated)

        self.vision_thread = threading.Thread(target=self.vision.run)
        self.vision_thread.start()

    def on_vision_updated(self, images):
        self.camera.setPixmap(images['camera'])
        self.edges.setPixmap(images['edges'])
        self.temp.setPixmap(images['temp'])

    def closeEvent(self, _):
        self.vision.stop()
        self.vision_thread.join()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.show()
    app.exec_()
