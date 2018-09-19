#!/usr/bin/env python2

import time
from abc import ABCMeta, abstractmethod

from PyQt5 import QtGui, QtCore
import tensorflow as tf
from object_detection.utils import ops as utils_ops

from naoqi import ALProxy
import numpy as np
import cv2


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
        return self.image


class ObjectDetection(object):
    def __init__(self, inference_graph):
        self.detection_graph = tf.Graph()
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
        m_v = -1
        m_i = 0
        for i, v in enumerate(results['detection_scores']):
            if v > m_v:
                m_v = v
                m_i = i
        return {
            'box': results['detection_boxes'][m_i],
            'score': m_v
        }


class Vision(QtCore.QObject):
    updated = QtCore.pyqtSignal(object)

    def __init__(self, image, inference_graph):
        super(Vision, self).__init__()
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        processed[:, :, 0] = clahe.apply(processed[:, :, 0])
        bgr = cv2.cvtColor(processed, cv2.COLOR_YUV2BGR)
        return cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)

    def edge_detection(self, img):
        # yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # yuv[:, :, 0] = cv2.Canny(yuv[:, :, 0], 100, 200)
        # return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        img = img.copy()
        for c in range(3):
            img[:, :, c] = cv2.Canny(img[:, :, c], 100, 200)
        return img

    def run(self):
        self._running = True
        last_time = 0
        while self._running:
            now = time.time()
            if now - last_time < 1/30.0:
                time.sleep((last_time + 1/30.0) - now)
            last_time = time.time()

            img = self.image.get_image()
            print(repr(self.object_detection.detect(img)))
            temp = self.postprocess(img)
            edges = self.edge_detection(temp)
            img = self.make_pixmap(img)
            edges = self.make_pixmap(edges)
            temp = self.make_pixmap(temp)
            self.updated.emit({
                'camera': img,
                'edges': edges,
                'temp': temp
            })

    def stop(self):
        self._running = False
