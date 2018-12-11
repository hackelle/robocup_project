import logging
from math import floor, ceil

import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops

MIN_SCORE = 0.3
# Part of a box that has to overlap with another to be considered intersecting
INTERSECTION_MIN = 0.2


class ObjectDetection(object):
    """Detect heads in the image using our Tensorflow model."""

    # TODO: Retrain NN for preprocessed images
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
                all_tensor_names = {output.name for op in ops
                                    for output in op.outputs}
                tensor_dict = {}
                for key in [
                        'num_detections', 'detection_boxes',
                        'detection_scores', 'detection_classes',
                        'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = (
                            tf.get_default_graph().get_tensor_by_name(
                                tensor_name
                            ))
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box
                    # coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0],
                                               [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                               [real_num_detection, -1, -1])
                    detection_masks_reframed = (
                        utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, image.shape[0],
                            image.shape[1]))
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                                detection_masks_reframed, 0)
                image_tensor = (
                    tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                )

                # Run inference
                output_dict = sess.run(
                    tensor_dict,
                    feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as
                # appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = (
                    output_dict['detection_boxes'][0])
                output_dict['detection_scores'] = (
                    output_dict['detection_scores'][0])
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = (
                        output_dict['detection_masks'][0])
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
