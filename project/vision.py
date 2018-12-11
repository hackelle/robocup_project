#!/usr/bin/env python2

import time
import logging
from threading import Lock, Condition
from collections import namedtuple

from PyQt5 import QtGui, QtCore

import cv2

from ellipse_detection import EllipseDetection
from geometry import GeometryCreation
from head_detection import ObjectDetection

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (2, 2)
CANNY_T1 = 120
CANNY_T2 = 180
# Factor we increase the bounding boxes by in each direction
BB_MULT = 1.25
FRAMERATE = 1/5.0


RobotFace = namedtuple("RobotFace", "eyes ear box")


class Vision(QtCore.QObject):
    """
    Main Vision hub.

    Links the GUI, head detection, ellipse detection and geometry creation as
    well as preprocessing the image in various ways.
    """

    updated = QtCore.pyqtSignal(object)

    def __init__(self, image, inference_graph):
        super(Vision, self).__init__()
        self.logger = logging.getLogger()
        self.image = image
        self._create_object_detection(inference_graph)
        self.condition = Condition()
        self.lock = Lock()
        self.paused = False
        self.display_next = False
        self._running = False

    def _create_object_detection(self, inference_graph):
        self.object_detection = ObjectDetection(inference_graph)

    def make_pixmap(self, cv2_img):
        height, width, _ = cv2_img.shape
        bytes_per_line = width * 3
        img = QtGui.QImage(cv2_img.data, width, height, bytes_per_line,
                           QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap(img)

    def preprocess(self, img):
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                tileGridSize=CLAHE_GRID_SIZE)
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = clahe.apply(processed)
        bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        return cv2.medianBlur(bgr, 3)

    def edge_detection(self, img):
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        edges = cv2.Canny(yuv[:, :, 0], CANNY_T1, CANNY_T2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def crop(self, img, box):
        box = list(map(float, box))
        dx = (box[2] - box[0]) / 2
        dy = (box[3] - box[1]) / 2
        mid_x = box[0] + dx
        mid_y = box[1] + dy
        height, width, _ = img.shape
        new_box = list(map(int, [
            max(0.0, mid_x - dx * BB_MULT),
            max(0.0, mid_y - dy * BB_MULT),
            min(width, mid_x + dx * BB_MULT),
            min(height, mid_y + dy * BB_MULT),
        ]))
        return img[new_box[1]:new_box[3], new_box[0]:new_box[2]].copy()

    def detect_heads(self, img):
        self.logger.debug("Running TF detection...")
        boxes = self.object_detection.detect(img)
        cropped = []
        processed = []
        edges = []
        faces = []
        for i, box in enumerate(boxes):
            cropped.append(self.crop(img, box['box']))
            self.logger.debug("Preprocessing Box #%i...", i)
            processed.append(self.preprocess(cropped[-1]))
            self.logger.debug("Detecting edges in Box #%i...", i)
            edges.append(self.edge_detection(processed[-1]))
            self.logger.debug("Detecting ellipses in Box #%i...", i)
            ellipse_detection = EllipseDetection(processed[-1], edges[-1])
            face = ellipse_detection.detect_ellipses()
            ellipse_detection.draw_ellipses(*face)
            faces.append(RobotFace(eyes=face[0], ear=face[1], box=box['box']))
            self.logger.debug("Done!")
        self.logger.debug("Creating Geometry...")
        geometry_creation = GeometryCreation(faces, img.shape[:2])
        geometry = geometry_creation.create()
        geometry_img = geometry_creation.draw(geometry)
        # TODO: Calculate the most likely angle, esp. if there are multiple
        # robots

        for box in boxes:
            b = box['box']
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0))

        return processed, edges, map(lambda t: t['score'], boxes), geometry_img

    def run(self):
        self._running = True
        last_time = 0
        while self._running:
            now = time.time()
            if now - last_time < 1/FRAMERATE:
                time.sleep((last_time + 1/FRAMERATE) - now)
            last_time = time.time()
            with self.condition:
                while self.paused:
                    if self.display_next:
                        self.display_next = False
                        break
                    self.condition.wait()

            self.logger.debug("Getting image...")
            with self.lock:
                img = self.image.get_image()
            self.logger.debug("Preprocessing...")
            img = self.preprocess(img)
            temp, edges, scores, geometry_img = self.detect_heads(img)
            geometry_img = self.make_pixmap(geometry_img)
            img = self.make_pixmap(img)
            edges = map(self.make_pixmap, edges)
            temp = map(self.make_pixmap, temp)
            self.updated.emit({
                'camera': img,
                'edges': edges,
                'temp': temp,
                'drawing': geometry_img,
                'scores': scores,
            })

    def pause(self):
        self.logger.info("Play/Pause")
        with self.condition:
            self.paused = not self.paused
            self.condition.notify_all()

    def prev(self):
        self.logger.info("Prev")
        with self.lock:
            self.image.prev()
            with self.condition:
                self.display_next = True
                self.condition.notify_all()

    def next(self):
        self.logger.info("Next")
        with self.lock:
            self.image.next()
            with self.condition:
                self.display_next = True
                self.condition.notify_all()

    def stop(self):
        self._running = False
