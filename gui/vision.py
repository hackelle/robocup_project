#!/usr/bin/env python2

import time
from abc import ABCMeta, abstractmethod

from PyQt5 import QtGui, QtCore

from naoqi import ALProxy
import numpy as np
import cv2


class ImageProvider(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_image(self):
        pass


class RCVisionProvider(ImageProvider):
    def __init__(self, address, camera_id=0):
        self.proxy = ALProxy('RobocupVision', address,
                             9559)
        self.camera_id = camera_id

    def get_image(self):
        data = self.proxy.getBGR24Image(self.camera_id)
        image = np.fromstring(data, dtype=np.uint8).reshape(
            (480, 640, 3)
        )
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class StorageVisionProvider(ImageProvider):
    def __init__(self, path):
        self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def get_image(self):
        return self.image


class Vision(QtCore.QObject):
    updated = QtCore.pyqtSignal(object)

    def __init__(self, image):
        super(Vision, self).__init__()
        self.image = image

    def rate_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 300)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # measure = cv2.Laplacian(gray, cv2.CV_64F).var()

        # cv2.putText(img, '{:.2f}'.format(measure), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    def make_pixmap(self, cv2_img):
        height, width, _ = cv2_img.shape
        bytes_per_line = width * 3
        img = QtGui.QImage(cv2_img.data, width, height, bytes_per_line,
                           QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap(img)

    def postprocess(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        processed[:, :, 0] = clahe.apply(processed[:, :, 0])
        bgr = cv2.cvtColor(processed, cv2.COLOR_YUV2BGR)
        return cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)

    def edge_detection(self, img):
        # yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # yuv[:, :, 0] = cv2.Canny(yuv[:, :, 0], 100, 200)
        # return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        img = img.copy()
        for c in range(3):
            img[:, :, c] = cv2.Canny(img[:, :, c], 100, 200)
        return img

    def run(self):
        self._running = True
        last_time = 0
        while self._running:
            now = time.time()
            if now - last_time < 1/30.0:
                time.sleep((last_time + 1/30.0) - now)
            last_time = time.time()

            img = self.image.get_image()
            temp = self.postprocess(img)
            edges = self.edge_detection(temp)
            img = self.make_pixmap(img)
            edges = self.make_pixmap(edges)
            temp = self.make_pixmap(temp)
            self.updated.emit({
                'camera': img,
                'edges': edges,
                'temp': temp
            })

    def stop(self):
        self._running = False
