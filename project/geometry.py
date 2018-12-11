import logging
from math import pi, sin

import cv2
import numpy as np

NAO_EAR_HEIGHT = 62.1  # mm
NAO_IMG_HEIGHT = 480  # px
NAO_SENSOR_HEIGHT = 968 * 1.9E-3  # mm
NAO_HFOV = 60.97 / 180 * pi
NAO_VFOV = 47.64 / 180 * np.pi
NAO_FOCAL_LENGTH = ((NAO_SENSOR_HEIGHT / 2) / sin(NAO_VFOV/2)
                    * sin(pi/2 - NAO_VFOV/2))
IMG_WIDTH = 480
IMG_HEIGHT = 480


class GeometryCreation(object):
    """Geometry calculation/creation based on detected eyes/ears"""

    def __init__(self, faces, img_shape):
        self.faces = faces
        self.img_shape = img_shape
        self.robots = []
        self.logger = logging.getLogger()

    def create(self):
        """
        Calculate/create the geometry.

        :return: A list of robots by position and orientation
        :rtype: list(tuple((x, y), theta))
        """
        geometry = []
        for i, face in enumerate(self.faces):
            if face.ear is None:
                self.logger.warn("Face %i has no ear, aborting!", i)
                continue

            height = face.ear[1][1]
            dist = (
                (NAO_FOCAL_LENGTH * NAO_EAR_HEIGHT * NAO_IMG_HEIGHT) /
                (height * NAO_SENSOR_HEIGHT)
            )
            radial_angle = self.calculate_radial_angle(face.box)
            self.logger.debug("Radial: {}, {}".format(dist, radial_angle))
            location = (
                dist * np.cos(radial_angle),
                dist * np.sin(radial_angle)
            )
            self.logger.debug("Cartesian: {}, {}".format(location[0],
                                                         location[1]))

            # Calculate angle (between [0, 180] deg)
            facing_angle, sure = self.calculate_facing_angle(face)
            self.logger.debug("local facing angle = {}".format(facing_angle))
            facing_angle = radial_angle - facing_angle

            geometry.append((location, facing_angle))

            self.logger.debug("facing angle = {}, sure = {}".format(
                facing_angle, sure))

        return geometry

    def calculate_radial_angle(self, box):
        rows = box[2] - box[0]
        x_center = (box[0] + rows / 2.0) / self.img_shape[0] - 0.5
        fov_angle = -x_center * NAO_HFOV / 2
        self.logger.debug("x_center={}\tfov_angle={}\tFOV={}".format(
            x_center, fov_angle, NAO_HFOV))
        return fov_angle + np.pi / 2

    def calculate_facing_angle(self, face):
        # This will return the angle from the (local!) x axis...
        angle = np.arccos(face.ear[1][0] / face.ear[1][1])
        self.logger.debug("angle before correction: {}".format(angle))
        # ... so subtract it from 90 deg
        angle = np.pi / 2 - angle

        if len(face.eyes) == 0:
            # The robot is looking away from us, so add 180 deg
            angle += np.pi

        rows = face.box[2] - face.box[0]
        ear_location = self.ear_location(face.ear, rows)
        if ear_location == 'right':
            angle = -angle
        sure = ear_location != 'middle'
        # TODO: Do something when we're unsure

        return angle, sure

    def ear_location(self, ear, rows):
        ear_x = ear[0][0]

        if ear_x < 2 * rows / 5.0:
            return 'left'
        elif ear_x > 4 * rows / 5.0:
            return 'right'
        else:
            return 'middle'

    def draw(self, geometry):
        """Draw the geometry into a new OpenCV image and return it."""
        img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), np.uint8)
        cv2.circle(img, (IMG_WIDTH / 2, IMG_HEIGHT), 11, (255, 0, 0), -1)

        for robot in geometry:
            location, facing = robot
            center = (
                int(round(IMG_WIDTH / 4000.0 * location[0] + IMG_WIDTH / 2.0)),
                IMG_HEIGHT - int(round(IMG_WIDTH / 4000.0 * location[1]))
            )
            self.logger.debug("center [px]: {}".format(center))
            cv2.circle(img, center, 5, (0, 255, 0), -1)

            fov_1 = (
                int(round(center[0] + 600 * np.cos(facing + NAO_HFOV / 2))),
                int(round(center[1] + 600 * np.sin(facing + NAO_HFOV / 2)))
            )
            fov_2 = (
                int(round(center[0] + 600 * np.cos(facing - NAO_HFOV / 2))),
                int(round(center[1] + 600 * np.sin(facing - NAO_HFOV / 2)))
            )
            cv2.line(img, center, fov_1, (0, 0, 255), 1)
            cv2.line(img, center, fov_2, (0, 0, 255), 1)
        return img
