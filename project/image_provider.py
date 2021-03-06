import logging
import os
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from naoqi import ALProxy


class ImageProvider(object):
    """Provides images from an abstract source"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_image(self):
        pass

    def next(self):
        pass

    def prev(self):
        pass


class RCVisionProvider(ImageProvider):
    """Image provider using the realtime images from the NAO."""

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
    """Image provider using one image stored locally."""

    def __init__(self, path):
        self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def get_image(self):
        return self.image.copy()


class DirectoryVisionProvider(ImageProvider):
    """Image provider using multiple images stored in a local directory."""

    def __init__(self, path):
        self.paths = filter(
            lambda p: os.path.isfile(p),
            map(
                lambda p: os.path.join(path, p),
                os.listdir(path)
            )
        )
        self.paths.sort()
        logging.info(repr(self.paths))
        self.images = map(
            lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
            self.paths
        )
        self.index = -1

    def get_image(self):
        self.index = (self.index + 1) % len(self.images)
        logging.info("Image: %s", self.paths[self.index])
        return self.images[self.index].copy()

    def next(self):
        self.index = (self.index + 1) % len(self.images)

    def prev(self):
        self.index = (self.index - 1) % len(self.images)
