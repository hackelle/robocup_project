import logging
import time
import random
from copy import copy
from math import sqrt, cos, pi, ceil
import skimage.measure

import cv2
import numpy as np

ANGLE_INCREMENT = 1


class EllipseDetection(object):
    def __init__(self, processed, edges):
        self.processed = processed
        self.edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        (self.rows, self.cols, _) = processed.shape
        self.head_area = self.rows*self.cols
        self.logger = logging.getLogger()
        self.edge_points = None

    def detect_ellipses(self):
        _, contours, _ = cv2.findContours(self.edges, 1, 2)
        # Flatten list
        self.edge_points = [point[0] for contour in contours for point in contour]
        self.edge_points = np.array(self.edge_points)
        contours = list(filter(lambda c: len(c) >= 5, contours))
        self.logger.debug("Post-filter: %d contours", len(contours))
        contours = self.merge_contours(contours)
        self.logger.debug("Post-merge: %d contours", len(contours))
        ellipses = self.filter_good_ellipses(contours)
        self.logger.debug("First filter: %d ellipses", len(ellipses))
        for ellipse in ellipses:
            cv2.ellipse(self.processed, ellipse, (255, 255, 0), 1)
        ellipses = self.filter_overlapping(ellipses)
        for ellipse in ellipses:
            cv2.ellipse(self.processed, ellipse, (0, 255, 255), 1)
        self.logger.debug("Second filter: %d ellipses", len(ellipses))
        return self.facial_structure(ellipses)

    def merge_contours(self, contours):
        MAX_DIST_TO_MERGE = 5

        contour_dist = self.contour_distances(contours)
        self.logger.debug("Distance matrix shape: %s", repr(contour_dist.shape))
        ret = copy(contours)

        for i, c1 in enumerate(contours):
            for j in range(i + 1, len(contours)):
                c2 = contours[j]
                if contour_dist[i, j] <= MAX_DIST_TO_MERGE:
                    ret.append(np.concatenate((c1, c2)))

        return ret

    def contour_distances(self, contours):
        dists = np.zeros((len(contours), len(contours)))
        for i, c1 in enumerate(contours):
            for j in range(i + 1, len(contours)):
                c2 = contours[j]
                dist = self.contour_distance(c1, c2)
                dists[i, j] = dists[j, i] = dist
        return dists

    def contour_distance(self, c1, c2):
        NTH_POINT = 3

        if len(c1) < len(c2):
            c1, c2, = c2, c1

        c2 = np.array(list(map(lambda p: p[0], c2)))
        min_dist = float('inf')
        for i in range(0, len(c1), NTH_POINT):
            p = tuple(c1[i][0])
            dist = abs(cv2.pointPolygonTest(c2, p, True))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def point_in_list(self, point, l):
        for p in l:
            if np.array_equal(p, point):
                return True
        return False

    def point_around_ellipse(self, point, ellipse, mult):
        """
        Check if a point is around an ellipse.

        :param point: The point
        :param ellipse: The ellipse
        :param mult: The multiplier difference which defines the threshold.
                     E.g. if mult=0.2, then we check if the point is outside
                     of an ellipse with a*=0.8, b*=0.8 and within an ellipse
                     with a*=1.2, b*=1.2
        """
        inner = list(ellipse)
        outer = list(ellipse)
        return (not self.point_in_ellipse(point, inner) and
                self.point_in_ellipse(point, outer))

    def point_in_ellipse(self, point, ellipse):
        # https://math.stackexchange.com/a/76463/101072
        point = self.rotate_point(point, -ellipse[2])
        x, y = point
        c_x, c_y = ellipse[0]
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        return ((x - c_x) ** 2 / a ** 2 + (y - c_y) ** 2 / b ** 2) <= 1

    def rotate_point(self, point, degrees):
        rad = degrees / 180 * np.pi
        c = np.cos(rad)
        s = np.sin(rad)
        x = point[0]
        y = point[1]
        return (
            c * x - s * y,
            s * x + c * y
        )

    def model_err(self, model, n_inliers):
        return self.ellipse_area(model) / n_inliers

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

        for i, c in enumerate(contours):
            if len(c) < 5:     # cant find ellipse with less
                self.logger.warning("Contour is too small: %d", i)
                continue
            ellipse = cv2.fitEllipseDirect(c)
            cv2.ellipse(self.processed, ellipse,
                        (255 - i * (255 / len(contours)), 0, 0), 1)
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
        big = filter(lambda t: self.ellipse_classify(t) == "big", ellipses)
        small = filter(lambda t: self.ellipse_classify(t) == "small", ellipses)

        ear = None
        if len(big) == 1:
            ear = big[0]

        if ear is not None:
            new_small = []
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
                eyes = {e}
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
            if minor / major < 0.8 and (60 < angle < 120):
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

        return self.check_partial_ellipse(ellipse)

    def check_partial_ellipse(self, ellipse):
        """
        Check if a contour fits a large enough part of the fitted ellipse.

        This means that at least 2/3 of the ellipse needs to have points on the
        contour "near" it.

        :param ellipse: The ellipse that was fitted onto `contour_points`
        :return Whether the contour fits a large enough part of the ellipse
        """

        x, y = ellipse[0]
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        phi = ellipse[2] * np.pi / 180.0
        outside_angle = 0
        min_dist = min(3, max(1.125, ceil(0.05 * max(a, b))))
        for angle in range(0, 360, ANGLE_INCREMENT):
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
            dist = abs(cv2.pointPolygonTest(self.edge_points, p, True))
            if dist > min_dist:
                outside_angle += ANGLE_INCREMENT

        cls = self.ellipse_classify(ellipse)
        max_outside = 8 if cls == 'small' else 16
        return outside_angle <= 360 / max_outside
