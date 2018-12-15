import logging
from math import pi, sin
from functools import reduce

import cv2
import numpy as np

NAO_EAR_HEIGHT = 62.1  # mm
NAO_EYE_DIST = 51.7  # mm
NAO_IMG_WIDTH = 640  # px
NAO_IMG_HEIGHT = 480  # px
NAO_SENSOR_WIDTH = 1288 * 1.9E-3  # mm
NAO_SENSOR_HEIGHT = 968 * 1.9E-3  # mm
NAO_HFOV = 60.97 / 180 * pi
NAO_VFOV = 47.64 / 180 * np.pi
NAO_FOCAL_LENGTH = ((NAO_SENSOR_HEIGHT / 2) / sin(NAO_VFOV/2)
                    * sin(pi/2 - NAO_VFOV/2))
ROBOCUP_MAX_DIST = 9000  # mm
IMG_WIDTH = 480
IMG_HEIGHT = 480

EAR_CENTER_FRACTION = 21.0


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

        :return: A list of robots by position and orientation and a list of
                 polygons
        :rtype: list(tuple((x, y), theta)), list(np.array)
        """
        geometry = []
        for i, face in enumerate(self.faces):
            if face.ear is not None:
                height = face.ear[1][1]
                dist = (
                    (NAO_FOCAL_LENGTH * NAO_EAR_HEIGHT * NAO_IMG_HEIGHT) /
                    (height * NAO_SENSOR_HEIGHT)
                )
            elif len(face.eyes) == 2:
                # x coord = [eye_i][0 = center][0 = x]
                eye_dist = abs(face.eyes[0][0][0] - face.eyes[1][0][0])
                dist = (
                    NAO_FOCAL_LENGTH * NAO_EYE_DIST * NAO_IMG_WIDTH /
                    (eye_dist * NAO_SENSOR_WIDTH)
                )
            else:
                self.logger.warn(
                    "Face %i has not enough information, aborting!", i
                )
                continue

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
            if sure:
                facing_angle = radial_angle - facing_angle
                geometry.append((location, facing_angle))
            else:
                self.logger.warn("Unsure about orientation of face %i", i)

        fovs = self.create_fovs(geometry)

        return geometry, fovs

    def create_fovs(self, geometry):
        fovs = []
        for robot in geometry:
            fovs.append(self.make_fov(robot))

        return fovs

    def make_fov(self, robot):
        location, facing = robot
        location = np.array(location)
        fov1 = np.array([
            ROBOCUP_MAX_DIST * np.cos(facing + NAO_HFOV / 2),
            -ROBOCUP_MAX_DIST * np.sin(facing + NAO_HFOV / 2),
        ]) + location
        fov2 = np.array([
            ROBOCUP_MAX_DIST * np.cos(facing - NAO_HFOV / 2),
            -ROBOCUP_MAX_DIST * np.sin(facing - NAO_HFOV / 2),
        ]) + location

        return np.array([fov1, fov2])

    def calculate_radial_angle(self, box):
        rows = box[2] - box[0]
        x_center = (box[0] + rows / 2.0) / self.img_shape[0] - 0.5
        fov_angle = -x_center * NAO_HFOV / 2
        self.logger.debug("x_center={}\tfov_angle={}\tFOV={}".format(
            x_center, fov_angle, NAO_HFOV))
        return fov_angle + np.pi / 2

    def calculate_facing_angle(self, face):
        if face.ear is None:
            if len(face.eyes) != 2:
                self.logger.error("No ear and no 2 eyes!?")
                return 0, False
            return 0, True

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

        return angle, sure

    def ear_location(self, ear, rows):
        ear_x = ear[0][0]

        left_part = EAR_CENTER_FRACTION // 2
        right_part = left_part + 1

        if ear_x <= left_part * rows / EAR_CENTER_FRACTION:
            return 'left'
        elif ear_x >= right_part * rows / EAR_CENTER_FRACTION:
            return 'right'
        else:
            return 'middle'

    def draw(self, geometry):
        """Draw the geometry into a new OpenCV image and return it."""
        DRAWING_DIM = 6000.0

        geometry, fovs = geometry
        img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), np.uint8)

        fov_drawings = []

        for i, robot in enumerate(geometry):
            fov = fovs[i]
            location, facing = robot
            center = (
                int(round(
                    IMG_WIDTH / DRAWING_DIM * location[0] + IMG_WIDTH / 2.0
                )),
                IMG_HEIGHT / 2 - int(round(
                    IMG_WIDTH / DRAWING_DIM * location[1]
                ))
            )
            self.logger.debug("center [px]: {}".format(center))
            cv2.circle(img, center, 5, (0, 255, 0), -1)

            fov[0] *= IMG_WIDTH / DRAWING_DIM * np.array([1, -1])
            fov[0] += np.array([IMG_WIDTH, IMG_HEIGHT]) / 2.0
            fov[0] = np.rint(fov[0])
            fov[1] *= IMG_WIDTH / DRAWING_DIM * np.array([1, -1])
            fov[1] += np.array([IMG_WIDTH, IMG_HEIGHT]) / 2.0
            fov[1] = np.rint(fov[1])
            fov = fov.astype(int)

            cv2.line(img, center, tuple(fov[0]),
                     (0, 0, 255), 1)
            cv2.line(img, center, tuple(fov[1]),
                     (0, 0, 255), 1)

            fov_drawing = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), np.uint8)
            triangle_cnt = np.array([center, tuple(fov[0]), tuple(fov[1])])
            cv2.drawContours(fov_drawing, np.array([triangle_cnt]), 0,
                             (0, 255, 255), -1)
            fov_drawings.append(fov_drawing)

        if len(fov_drawings) > 0:
            fov_intersection = reduce(np.bitwise_and, fov_drawings, np.ones((
                IMG_WIDTH, IMG_HEIGHT, 3), np.uint8) * 255)
            img += fov_intersection

        cv2.circle(img, (IMG_WIDTH / 2, IMG_HEIGHT / 2), 11, (255, 0, 0), -1)

        # Draw concentric circles every meter
        for r in range(1000, 11000, 1000):
            r_px = int(round(r * IMG_WIDTH / DRAWING_DIM))
            cv2.circle(img, (IMG_WIDTH / 2, IMG_HEIGHT / 2), r_px,
                       (255, 255, 255), 1)
            cv2.putText(img, "{} m".format(r / 1000),
                        (IMG_WIDTH / 2 + r_px, IMG_HEIGHT / 2),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))

        return img
