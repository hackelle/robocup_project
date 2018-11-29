#!/usr/bin/env python2

import cv2
import sys


ADDRESS = '10.0.7.15'
CAMERA_ID = 0


def convert_img(path):
    img = cv2.imread(path, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("converted-" + path, rgb_img)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} IMG_PATH'.format(sys.argv[0]))
        sys.exit(63)
    convert_img(sys.argv[1])
