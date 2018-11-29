import logging
import os
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from naoqi import ALProxy


class ImageProvider(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_image(self):
        pass

    def next(self):
        pass

    def prev(self):
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
        return self.image.copy()


class DirectoryVisionProvider(ImageProvider):
    def __init__(self, path):
        self.images = filter(
            lambda p: os.path.isfile(p),
            map(
                lambda p: os.path.join(path, p),
                os.listdir(path)
            )
        )
        logging.info(repr(self.images))
        self.images = map(
            lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
            self.images
        )
        self.index = 0

    def get_image(self):
        self.index = (self.index + 1) % len(self.images)
        return self.images[self.index].copy()

    def next(self):
        self.index = (self.index + 1) % len(self.images)

    def prev(self):
        self.index = (self.index - 1) % len(self.images)
