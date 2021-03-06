import logging
import time
from copy import copy
from math import sqrt, cos, pi, ceil
from functools import reduce

import cv2
import numpy as np

ANGLE_INCREMENT = 1
EAR_MIN_SIZE = 0.25  # relative to head size
EAR_MAX_SIZE = 0.5   # relative to head size
EAR_MAX_OUTSIDE = 8  # fraction of circle that may not have edge pixels
EAR_MAX_ANGLE = 20   # degrees the ellipse may be rotated by
EYE_MIN_HEIGHT = 8   # px
EYE_MIN_WIDTH = 4    # px
EYE_MAX_SIZE = 0.15  # relative to head size
EYE_MAX_OUTSIDE = 8  # fraction of circle that may not have edge pixels
ELLIPSE_OVERLAP = 10 # pixel for overlaping ellipses


class EllipseDetection(object):
    """Ellipse detection and filtering"""

    def __init__(self, processed, edges):
        self.processed = processed
        self.edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        (self.rows, self.cols, _) = processed.shape
        self.head_area = self.rows*self.cols
        self.logger = logging.getLogger()
        self.edge_points = None

    def detect_ellipses(self):
        """
        Detect the eye/ear ellipses.

        :return: A list of recognized eyes and an ear (or None)
        :rtype: list, (ellipse or None)
        """
        _, contours, _ = cv2.findContours(self.edges, 1, 2)
        # Flatten list
        self.edge_points = [point[0] for contour in contours
                            for point in contour]
        self.edge_points = np.array(self.edge_points)

        # Contour filtering/merging
        contours = list(filter(lambda c: len(c) >= 5, contours))
        self.logger.debug("Post-filter: %d contours", len(contours))
        contours = self.merge_contours(contours)
        self.logger.debug("Post-merge: %d contours", len(contours))

        # Ellipse detection and first filter step
        ellipses = self.filter_good_ellipses(contours)
        self.logger.debug("First filter: %d ellipses", len(ellipses))
        # TODO: Generate some images for report and remove
        for ellipse in ellipses:
            cv2.ellipse(self.processed, ellipse, (255, 255, 0), 1)

        # Second filter step (merge overlapping ellipses)
        ellipses = self.filter_overlapping(ellipses)
        # TODO: Generate some images for report and remove
        for ellipse in ellipses:
            cv2.ellipse(self.processed, ellipse, (0, 255, 255), 1)
        self.logger.debug("Second filter: %d ellipses", len(ellipses))

        # Final filtering and classification
        return self.facial_structure(ellipses)

    def merge_contours(self, contours):
        MAX_DIST_TO_MERGE = 5

        contour_dist = self.contour_distances(contours)
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

    def draw_ellipses(self, eyes, ear):
        """Draw the ellipses into the (existing) processed image."""
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
        i = 0
        while i < len(ellipses) - 1:
            current_ellipse = ellipses[i]
            candidates = [ellipses[i]]

            for ellipse in ellipses[i+1:]:
                if self.ellipses_overlapping(current_ellipse, ellipse):
                    self.logger.debug("Ellipses overlap!")
                    candidates.append(ellipse)

            if len(candidates) > 1:
                max_e = max(candidates, key=self.ellipse_area)
                i -= 1
                for e in candidates:
                    if e != max_e:
                        ellipses.remove(e)

            i += 1

        return ellipses

    def ellipses_overlapping(self, e1, e2):
        d1 = np.zeros((self.rows, self.cols, 3), np.uint8)
        self.logger.debug(repr(e1))
        cv2.ellipse(d1, e1, (1, 0, 0), -1)
        d2 = np.zeros((self.rows, self.cols, 3), np.uint8)
        cv2.ellipse(d2, e2, (1, 0, 0), -1)
        intersection = np.bitwise_and(d1, d2)
        overlap = sum(intersection.flatten())
        return overlap > ELLIPSE_OVERLAP

    def ellipse_area(self, ellipse):
        return np.pi * ellipse[1][0] * ellipse[1][1] / 4

    def ellipse_x_coords(self, ellipse):
        x_dim = abs(ellipse[1][1] * cos(ellipse[2] / 180 * pi - pi/2))
        x_dim = max(x_dim, ellipse[1][0])

        return (ellipse[0][0] - x_dim / 2,
                ellipse[0][0] + x_dim / 2)

    def ellipse_y_coords(self, ellipse):
        y_dim = abs(ellipse[1][1] * cos(ellipse[2] / 180 * pi))
        y_dim = max(y_dim, ellipse[1][0])
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
        self.logger.debug(repr(ellipse))
        height = ellipse[1][1] / self.rows

        if EAR_MIN_SIZE <= height <= EAR_MAX_SIZE:
            self.logger.debug("Big ellipse!")
            return "big"
        elif (EYE_MIN_WIDTH <= ellipse[1][0] and
              EYE_MIN_HEIGHT <= ellipse[1][1] and height <= EYE_MAX_SIZE):
            return "small"
        else:
            return "invalid"

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
            ear_x = self.ellipse_x_coords(ear)
            ear_y = self.ellipse_y_coords(ear)
            for e in small:
                # Check that the eye doesn't overlap with the ear
                if ear_x[0] < e[0][0] < ear_x[1]:
                    self.logger.info("Small ellipse at same x coords as ear")
                    continue

                # Check that the eye isn't above or below the ear
                self.logger.info("Eye is at %f, ear at [%f, %f]", e[0][1],
                                 ear_y[0], ear_y[1])
                self.logger.debug(repr(ear))
                self.logger.debug(repr(e))
                if ear_y[0] < e[0][1] < ear_y[1]:
                    new_small.append(e)
                else:
                    self.logger.info("Small ellipse above/below ear")
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

        if minor / major < 0.1:
            return False

        # We're only interested in eyes or ears, so we filter out all ellipses
        # that can't be either
        if e_class == "big":
            # Big ellipses could be ears
            if minor / major < 0.8 and \
               (90 - EAR_MAX_ANGLE < angle < 90 + EAR_MAX_ANGLE):
                self.logger.debug("Rotated, very elongated")
                # Rotated and very elongated
                return False

            if r_center > self.rows * 0.75 or r_center < self.rows * 0.3:
                self.logger.debug("Too high/low")
                # Too high/low on the head
                return False
        elif e_class == "small":
            # Small ellipses could be eyes
            if minor / major < 0.3 and (45 < angle < 135):
                # Rotated and very elongated
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
        max_outside = EYE_MAX_OUTSIDE if cls == 'small' else EAR_MAX_OUTSIDE
        return outside_angle <= 360 / max_outside
