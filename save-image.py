#!/usr/bin/env python2

from naoqi import ALProxy
import cv2
import numpy as np
import sys


ADDRESS = '10.0.7.14'
CAMERA_ID = 0


def save_img(proxy, path):
    data = proxy.getBGR24Image(CAMERA_ID)
    img = np.fromstring(data, dtype=np.uint8).reshape(
        (480, 640, 3)
    )
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, rgb_img)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} IMG_PATH'.format(sys.argv[0]))
        sys.exit(63)
    proxy = ALProxy('RobocupVision', ADDRESS, 9559)
    save_img(proxy, sys.argv[1])
