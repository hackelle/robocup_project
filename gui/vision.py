#!/usr/bin/env python2

import time
from abc import ABCMeta, abstractmethod
from math import floor, ceil, sqrt, sin, cos, pi
import logging
import os
from threading import Lock, Condition

from PyQt5 import QtGui, QtCore
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from copy import copy

from naoqi import ALProxy
import numpy as np
import cv2


MIN_SCORE = 0.3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (2, 2)
CANNY_T1 = 100
CANNY_T2 = 200
# Part of a box that has to overlap with another to be considered intersecting
INTERSECTION_MIN = 0.2
# Factor we increase the bounding boxes by in each direction
BB_MULT = 1.25
FRAMERATE = 1/5.0


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


class ObjectDetection(object):
    def __init__(self, inference_graph):
        self.detection_graph = tf.Graph()
        self.logger = logging.getLogger()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(inference_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                                detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def detect(self, image):
        results = self.run_inference_for_single_image(image)
        boxes = []
        for i, v in enumerate(results['detection_scores']):
            if v < MIN_SCORE:
                break
            height, width, _ = image.shape
            box = results['detection_boxes'][i]
            box = [
                int(floor(box[1] * width)),
                int(floor(box[0] * height)),
                int(ceil(box[3] * width)),
                int(ceil(box[2] * height)),
            ]

            if self.intersects(box, map(lambda b: b['box'], boxes)):
                break

            boxes.append({
                'box': box,
                'score': v
            })
        return boxes

    def intersects(self, box, others):
        for other in others:
            dx = min(other[2], box[2]) - max(other[0], box[0])
            dy = min(other[3], box[3]) - max(other[2], box[2])
            if dx >= 0 and dy >= 0:
                intersection = dx * dy
                area = (box[2] - box[0]) * (box[3] - box[1])
                return float(intersection) / area > INTERSECTION_MIN
        return False


class Vision(QtCore.QObject):
    updated = QtCore.pyqtSignal(object)

    def __init__(self, image, inference_graph):
        super(Vision, self).__init__()
        self.logger = logging.getLogger()
        self.image = image
        self.create_object_detection(inference_graph)
        self.condition = Condition()
        self.lock = Lock()
        self.paused = False
        self.display_next = False

    def create_object_detection(self, inference_graph):
        self.object_detection = ObjectDetection(inference_graph)

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
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                tileGridSize=CLAHE_GRID_SIZE)
        # processed = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # processed[:, :, 0] = clahe.apply(processed[:, :, 0])
        # bgr = cv2.cvtColor(processed, cv2.COLOR_YUV2BGR)
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = clahe.apply(processed)
        bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        # return cv2.blur(bgr, (2, 2))
        return cv2.medianBlur(bgr, 3)
        # return cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)

    def edge_detection(self, img):
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        edges = cv2.Canny(yuv[:, :, 0], 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.Canny(gray, 100, 200)
        # return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # img = img.copy()
        # for c in range(3):
        #     img[:, :, c] = cv2.Canny(img[:, :, c], CANNY_T1, CANNY_T2)
        # return img

    def crop(self, img, box):
        box = list(map(float, box))
        dx = (box[2] - box[0]) / 2
        dy = (box[3] - box[1]) / 2
        mid_x = box[0] + dx
        mid_y = box[1] + dy
        height, width, _ = img.shape
        new_box = list(map(int, [
            max(0, mid_x - dx * BB_MULT),
            max(0, mid_y - dy * BB_MULT),
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
            self.logger.debug("Postprocessing Box #%i...", i)
            processed.append(self.postprocess(cropped[-1]))
            self.logger.debug("Detecting edges in Box #%i...", i)
            edges.append(self.edge_detection(processed[-1]))
            self.logger.debug("Detecting ellipses in Box #%i...", i)
            ellipse_detection = EllipseDetection(processed[-1], edges[-1])
            faces.append(ellipse_detection.detect_ellipses())
            ellipse_detection.draw_ellipses(*faces[-1])
            self.logger.debug("Done!")
        self.logger.debug("Creating Geometry...")
        geometry_creation = GeometryCreation(faces)
        geometry = geometry_creation.create()
        geometry_img = geometry_creation.draw(geometry)

        for box in boxes:
            b = box['box']
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0))

        return processed, edges, map(lambda b: b['score'], boxes), geometry_img

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
            self.logger.debug("Postprocessing...")
            img = self.postprocess(img)
            # temp = self.postprocess(img)
            # edges = self.edge_detection(temp)
            temp, edges, scores, geometry_img = self.detect_heads(img)
            geometry_img = self.make_pixmap(img)
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


class EllipseDetection(object):
    def __init__(self, processed, edges):
        self.processed = processed
        self.edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        (self.rows, self.cols, _) = processed.shape
        self.head_area = self.rows*self.cols
        self.logger = logging.getLogger()

    def detect_ellipses(self):
        _, contours, _ = cv2.findContours(self.edges, 1, 2)
        # Flatten list
        self.edge_points = [point for contour in contours for point in contour]
        self.edge_points = np.array(self.edge_points)
        ellipses = self.filter_good_ellipses(contours)
        ellipses = self.filter_overlapping(ellipses)
        return self.facial_structure(ellipses)

    def draw_ellipses(self, eyes, ear):
        for eye in eyes:
            cv2.ellipse(self.processed, eye, (0, 255, 0), 1)
        if ear is not None:
            cv2.ellipse(self.processed, ear, (0, 0, 255), 1)

    def filter_good_ellipses(self, contours):
        """
        Filter out all bad ellipses and return the good ones.

        :param contours: The contours in the image
        :return: List of good ellipses
        :rtype: list
        """
        good = []

        for c in contours:
            if len(c) < 5:     # cant find ellipse with less
                continue
            ellipse = cv2.fitEllipse(c)
            if self.check_ellipse(ellipse, c):
                good.append(copy(ellipse))

        return good

    def filter_overlapping(self, ellipses):
        """
        Filter ellipses that overlap and return the larger ones.

        :param ellipses: The ellipses
        :return: List of filtered ellipses
        :rtype: list
        """
        max_distance = 0.05 * sqrt(self.head_area)
        i = 0
        while i < len(ellipses) - 1:
            center = ellipses[i][0]
            candidates = [ellipses[i]]

            for ellipse in ellipses[i+1:]:
                c = ellipse[0]
                distance = sqrt((center[0] - c[0])**2 + (center[1] - c[1])**2)
                if distance <= max_distance:
                    candidates.append(ellipse)

            time.sleep(0.01)
            if len(candidates) > 1:
                max_e = max(candidates, key=self.ellipse_area)
                i -= 1
                for e in candidates:
                    if e != max_e:
                        ellipses.remove(e)

            i += 1

        return ellipses

    def ellipse_area(self, ellipse):
        return np.pi * ellipse[1][0] * ellipse[1][1] / 4

    def ellipse_y_coords(self, ellipse):
        y_dim = ellipse[1][1] * cos(ellipse[2] / 180 * pi)

        return (ellipse[0][1] - y_dim / 2,
                ellipse[0][1] + y_dim / 2)

    def ellipse_classify(self, ellipse, n_points=None):
        """
        Classify an ellipse by its size

        :param ellipse: Ellipse to classify
        :param n_points: Number of points in the contour for the ellipse. If
                         this is None, we won't check against ellipses that we
                         can't be sure about
        """
        area = self.ellipse_area(ellipse)
        if area < 1:
            return "very small"
        elif n_points is not None and n_points / area <= 0.02:
            return "unsure"
        elif 0.2 > area / self.head_area > 0.03:
            return "big"
        elif 0.03 >= area / self.head_area > 0.0025:
            return "small"
        else:
            return "wrong"

    def facial_structure(self, ellipses):
        """
        Splits ellipses into eyes and an ear and filters unreasonable
        configurations.

        :return: A list of recognized eyes and an ear (or None)
        :rtype: list, (ellipse or None)
        """
        big = filter(lambda e: self.ellipse_classify(e) == "big", ellipses)
        small = filter(lambda e: self.ellipse_classify(e) == "small", ellipses)

        ear = None
        if len(big) == 1:
            ear = big[0]

        new_small = []
        if ear is not None:
            ear_y = self.ellipse_y_coords(ear)
            for e in small:
                # Check that the eye isn't above or below the ear
                self.logger.info("Eye is at %f, ear at [%f, %f]", e[0][1],
                                 ear_y[0], ear_y[1])
                if ear_y[0] < e[0][1] < ear_y[1]:
                    new_small.append(e)
                else:
                    self.logger.warn("Eye above/below ear")
        small = new_small

        if len(small) < 2:
            # One eye/no eyes
            return small, ear
        else:
            # Check if there are only two eyes on the same height
            eye_candidates = set()
            max_dist = 0.1 * sqrt(self.head_area)

            for i, e in enumerate(small):
                # Find eyes on the same height as e
                eyes = set([e])
                center_row = e[0][1]
                left = None
                if ear is not None:
                    left = e[0][0] < ear[0][0]

                for other in small[i+1:]:
                    # The eyes can't be on opposite sides of the ear
                    if left is not None:
                        if (left and other[0][0] > ear[0][0]) or \
                           (not left and other[0][0] < ear[0][0]):
                            continue

                    if abs(other[0][1] - center_row) < max_dist:
                        eyes.add(other)
                if len(eyes) == 2:
                    eye_candidates.add(frozenset(eyes))

            if len(eye_candidates) == 1:
                return list(eye_candidates.pop()), ear
            else:
                return [], ear

    def check_ellipse(self, ellipse, contour_points):
        """
        Checks whether this ellipse should be considered for further eye/ear
        detection or not

        :param ellipse: checked ellipse
        :param contour_points: points in the contour
        :return: whether this ellipse should be considered for further eye/ear
                detection
        :rtype: bool
        """

        e_class = self.ellipse_classify(ellipse, len(contour_points))
        c_center = ellipse[0][0]
        r_center = ellipse[0][1]
        minor = ellipse[1][0]
        major = ellipse[1][1]
        angle = ellipse[2]

        # We're only interested in eyes or ears, so we filter out all ellipses
        # that can't be either
        if e_class == "very small":
            return False
        elif e_class == "unsure":
            return False
        elif e_class == "big":
            # Big ellipses could be ears
            if minor / major < 0.6 and (45 < angle < 135):
                # Rotated and very elongated
                return False

            if r_center > self.rows * 0.75 or r_center < self.rows * 0.3:
                # Too high/low on the head
                return False
        elif e_class == "small":
            # Small ellipses could be eyes
            if minor / major < 0.3 and (45 < angle < 135):
                # Rotate and very elongated
                return False

            if r_center > self.rows * 0.8 or r_center < self.rows * 0.4:
                # Too high/low on the head
                return False

            if c_center > self.cols * 0.9 or c_center < self.cols * 0.1:
                # Too far right/left on the head
                return False
        else:
            return False

        return self.check_partial_ellipse(ellipse, contour_points)

    def check_partial_ellipse(self, ellipse, contour_points):
        """
        Check if a contour fits a large enough part of the fitted ellipse.

        This means that at least 2/3 of the ellipse needs to have points on the
        contour "near" it.

        :param ellipse: The ellipse that was fitted onto `contour_points`
        :param contour_points: The contour
        :return Whether the contour fits a large enough part of the ellipse
        """
        ANGLE = 5
        x, y = ellipse[0]
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        phi = ellipse[2] * np.pi / 180.0
        outside_angle = 0
        min_dist = max(1.125, ceil(0.05 * max(a, b)))
        for angle in range(0, 360, ANGLE):
            rad = angle * np.pi / 180
            # Circle parametrisation
            p = (
                np.cos(rad),
                np.sin(rad)
            )
            # Scaling
            p = (
                a * p[0],
                b * p[1]
            )
            # Rotation
            p = (
                p[0] * np.cos(phi) - p[1] * np.sin(phi),
                p[0] * np.sin(phi) + p[1] * np.cos(phi)
            )
            # Translation
            p = (
                p[0] + x,
                p[1] + y
            )
            p1 = map(lambda i: int(round(i)), p)
            p1 = (p1[0], p1[1])
            dist = abs(cv2.pointPolygonTest(self.edge_points, p, True))
            if dist > min_dist:
                outside_angle += ANGLE

        return outside_angle <= 360 / 8


NAO_EAR_HEIGHT = 62.1  # mm
NAO_IMG_HEIGHT = 480  # px
NAO_SENSOR_HEIGHT = 968 * 1.9E-3  # mm
NAO_VFOV = 47.64 / 180 * pi
NAO_FOCAL_LENGTH = ((NAO_SENSOR_HEIGHT / 2) / sin(NAO_VFOV/2)
                    * sin(pi/2 - NAO_VFOV/2))


class GeometryCreation(object):
    def __init__(self, faces):
        self.faces = faces
        self.robots = []
        self.logger = logging.getLogger()

    def create(self):
        for i, face in enumerate(self.faces):
            eyes, ear = face
            if ear is None:
                self.logger.warn("Face %i has no ear, aborting!", i)
                continue

            height = ear[1][1]
            dist = (
                (NAO_FOCAL_LENGTH * NAO_EAR_HEIGHT * NAO_IMG_HEIGHT) /
                (height * NAO_SENSOR_HEIGHT)
            )
            angle = np.arccos(ear[1][0] / ear[1][1])
            self.logger.debug("d = {}, angle = {}".format(dist, angle))

    def draw(self, geometry):
        pass
