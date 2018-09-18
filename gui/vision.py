#!/usr/bin/env python2

import time
from abc import ABCMeta, abstractmethod

import gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import GLib

from naoqi import ALProxy
import numpy as np
import matplotlib.pyplot as plot
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


class Vision(object):
    def __init__(self, gui, image):
        self.gui = gui
        self.image = image

    def rate_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 300)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # measure = cv2.Laplacian(gray, cv2.CV_64F).var()

        # cv2.putText(img, '{:.2f}'.format(measure), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # def get_image(self, camera_id=0):
        # rgb_img = self.rate_image(rgb_img)
        # gbytes = GLib.Bytes.new(rgb_img.tostring())

        # return GdkPixbuf.Pixbuf.new_from_bytes(
        #     gbytes,
        #     GdkPixbuf.Colorspace.RGB,
        #     False,
        #     8,
        #     640,
        #     480,
        #     640 * 3,
        # )

    def make_pixbuf(self, cv2_img):
        gbytes = GLib.Bytes.new(cv2_img.tostring())

        return GdkPixbuf.Pixbuf.new_from_bytes(
            gbytes,
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            640,
            480,
            640 * 3,
        )

    def run(self):
        self._running = True
        last_time = 0
        while self._running:
            # now = time.time()
            # if now - last_time < 1/30.0:
            #     time.sleep((last_time + 1/30.0) - now)
            # last_time = time.time()
            time.sleep(0.1)

            img = self.image.get_image()
            temp = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            temp[:, :, 0] = cv2.equalizeHist(temp[:, :, 0])
            temp = self.make_pixbuf(cv2.cvtColor(temp, cv2.COLOR_YUV2BGR))
            img = self.make_pixbuf(img)
            self.gui.update_images({'camera': img, 'temp': temp})

    def stop(self):
        self._running = False
