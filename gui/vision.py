#!/usr/bin/env python2

import time
from abc import ABCMeta, abstractmethod
from math import floor, ceil, sqrt
import logging
import os

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
        for i, box in enumerate(boxes):
            cropped.append(self.crop(img, box['box']))
            self.logger.debug("Postprocessing Box #%i...", i)
            processed.append(self.postprocess(cropped[-1]))
            self.logger.debug("Detecting edges in Box #%i...", i)
            edges.append(self.edge_detection(processed[-1]))
            self.logger.debug("Detecting ellipses in Box #%i...", i)
            ellipse_detection = EllipseDetection(processed[-1], edges[-1])
            ellipse_detection.draw_ellipses()

        for box in boxes:
            b = box['box']
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0))

        return processed, edges, map(lambda b: b['score'], boxes)

    def run(self):
        self._running = True
        last_time = 0
        while self._running:
            now = time.time()
            if now - last_time < 1/FRAMERATE:
                time.sleep((last_time + 1/FRAMERATE) - now)
            last_time = time.time()

            self.logger.debug("Getting image...")
            img = self.image.get_image()
            self.logger.debug("Postprocessing...")
            img = self.postprocess(img)
            # temp = self.postprocess(img)
            # edges = self.edge_detection(temp)
            temp, edges, scores = self.detect_heads(img)
            img = self.make_pixmap(img)
            edges = map(self.make_pixmap, edges)
            temp = map(self.make_pixmap, temp)
            self.updated.emit({
                'camera': img,
                'edges': edges,
                'temp': temp,
                'scores': scores,
            })

    def stop(self):
        self._running = False


class EllipseDetection(object):
    def __init__(self, processed, edges):
        self.processed = processed
        self.edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        (self.rows, self.cols, _) = processed.shape
        self.head_area = self.rows*self.cols

    def draw_ellipses(self):
        candidates = []
        _, contours, _ = cv2.findContours(self.edges, 1, 2)
        max_area = -1
        max_ellipse = None
        ellipses = self.filter_good_ellipses(contours)
        ellipses = self.filter_overlapping(ellipses)

        for ellipse in ellipses:
            area = self.ellipse_area(ellipse)
            if area > max_area:
                max_area = area
                max_ellipse = ellipse

            # cv2.ellipse(processed, ellipse, (0, i * 255 / len(contours), 0), 1)
            cv2.ellipse(self.processed, ellipse, (0, 255, 0), 1)
            center = map(round, ellipse[0])
            center = (int(center[0]), int(center[1]))
            cv2.rectangle(self.processed, center, center, (255, 255, 0), 1)

            for e in candidates:
                if self.check_strong_ellipse_intersection(ellipse, e):
                    # cv2.ellipse(processed, ellipse, (0, 0, i * 255 / len(contours) + 10), 1)
                    # cv2.ellipse(processed, e, (0, 0, i * 255 / len(contours) + 10), 1)
                    cv2.ellipse(self.processed, ellipse, (0, 0, 255), 1)
                    cv2.ellipse(self.processed, e, (0, 0, 255), 1)

            candidates.append(copy(ellipse))

        if max_ellipse is not None:
            cv2.ellipse(self.processed, max_ellipse, (255, 0, 0), 1)

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

        return good

    def filter_overlapping(self, ellipses):
        """
        Filter ellipses that overlap and return the larger ones.

        :param ellipses: The ellipses
        :return: List of filtered ellipses
        :rtype: list
        """
        min_distance = 0.05 * sqrt(self.head_area)
        i = 0
        while i < len(ellipses) - 1:
            center = ellipses[i][0]
            candidates = [ellipses[i]]

            for ellipse in ellipses[i+1:]:
                c = ellipse[0]
                distance = sqrt((center[0] - c[0])**2 + (center[1] - c[1])**2)
                if distance <= min_distance:
                    candidates.append(ellipse)

            if len(candidates) > 1:
                max_e = max(candidates, key=self.ellipse_area)
                i -= len(candidates) - 1
                for e in candidates:
                    if e != max_e:
                        ellipses.remove(e)

            i += 1

        return ellipses

    def ellipse_area(self, ellipse):
        return np.pi * ellipse[1][0] * ellipse[1][1] / 4

    def check_ellipse(self, ellipse, contour_points):
        """
        Checks whether this ellipse should be considered for further eye/ear
        detection or not

        :param ellipse: checked ellipse
        :param head_area: total area of the head
        :param rows: number of rows
        :param cols: number of columns
        :param contour_points: points in the contour
        :return: whether this ellipse should be considered for further eye/ear
                detection
        :rtype: bool
        """

        area = np.pi * ellipse[1][0] * ellipse[1][1] / 4  # Formula for Ellipse Area for full axes
        c_center = ellipse[0][0]
        r_center = ellipse[0][1]
        minor = ellipse[1][0]
        major = ellipse[1][1]
        angle = ellipse[2]

        if area < 1:   # very small
            return False

        if float(len(contour_points)) / area <= 0.02:  # very unsure
            return False

        if 0.2 > area / self.head_area > 0.03:
            # big possible ellipse

            if minor / major < 0.6 and (45 < angle < 135):
                return False

            if r_center > self.rows * 0.75 or r_center < self.rows * 0.3:
                return False

        elif 0.03 >= area / self.head_area > 0.0025:
            # small possible ellipse
            return False

            if minor / major < 0.3 and (45 < angle < 135):
                return False

            if r_center > self.rows * 0.8 or r_center < self.rows * 0.4:
                return False

            if c_center > self.cols * 0.9 or c_center < self.cols * 0.1:
                return False

        else:
            # ellipse has wrong size (very small/big)
            return False

        self.check_partial_ellipse(ellipse, contour_points)
        return True

    def check_partial_ellipse(self, ellipse, contour_points):
        """
        Check if a contour fits a large enough part of the fitted ellipse.

        :param ellipse: The ellipse that was fitted onto `contour_points`
        :param contour_points: The contour
        :return Whether the contour fits a large enough part of the ellipse
        """
        x, y = ellipse[0]
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        # a, b = ellipse[1]
        phi = -(ellipse[2] * np.pi / 180.0)
        outside_angle = 0
        min_dist = 0.05 * max(a, b)
        logging.debug("a=%f, b=%f, phi=%f", a, b, phi)
        for angle in range(0, 360, 15):
            rad = angle * np.pi / 180
            p = (a * np.cos(rad), b * np.sin(rad))
            p = (
                p[0] * np.cos(phi) - p[1] * np.sin(phi) + x,
                p[0] * np.sin(phi) - p[1] * np.cos(phi) + y
            )
            # p = (
            #     a * np.cos(rad) * np.cos(phi) - b * np.sin(rad) * np.sin(phi) + x,
            #     a * np.cos(rad) * np.sin(phi) + b * np.sin(rad) * np.cos(phi) + y
            # )
            p1 = map(lambda i: int(round(i)), p)
            p1 = (p1[0], p1[1])
            cv2.rectangle(self.processed, p1, p1, (0, 255, 255), 1)
            dist = abs(cv2.pointPolygonTest(contour_points, p, True))
            if dist > min_dist:
                # logging.debug("dist=%f, min_dist=%f", dist, min_dist)
                outside_angle += 15

        return outside_angle <= 360

    def check_strong_ellipse_intersection(self, ellipse1, ellipse2):
        """
        Checks whether there is a strong intersection between these ellipses
        by checking whether the center point of one is inside the other

        :return: whether there is a strong intersection between these ellipses
        :rtype: bool
        """

        return self.check_point_in_ellipse(ellipse1[0][0], ellipse1[0][1], ellipse2[0][0], ellipse2[0][1], ellipse1[1][1], ellipse1[1][0], ellipse1[2]) or \
                    self.check_point_in_ellipse(ellipse2[0][0], ellipse2[0][1], ellipse1[0][0], ellipse1[0][1], ellipse2[1][1], ellipse2[1][0], ellipse2[2])

    def check_point_in_ellipse(self, c_ell, r_ell, c_p, r_p, major, minor, angle):
        """
        Checks whether the point is inside the ellipse or not

        :param c_ell: column of center point of the ellipse
        :param r_ell: row of center point of the ellipse
        :param c_p: column of the point
        :param r_p: row of the point
        :param major: mayor axis of the ellipse
        :param minor: minor axis of the ellipse
        :param angle: angle of the ellipse
        :return: whether it is inside or not
        :rtype: bool
        """

        c_rotated = c_ell + (c_p - c_ell) * np.cos(-angle) - (r_p - r_ell) * np.sin(-angle)
        r_rotated = r_ell + (c_p - c_ell) * np.sin(-angle) + (r_p - r_ell) * np.cos(-angle)

        return ((c_rotated - c_ell) ** 2) / ((minor / 2) ** 2) + ((r_rotated - r_ell) ** 2) / ((major / 2) ** 2) <= 1
