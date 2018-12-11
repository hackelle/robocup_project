#!/usr/bin/env python2

import numpy as np
import cv2
import tensorflow as tf

from object_detection.utils import ops as utils_ops

PATH_TO_FROZEN_GRAPH = (
    '/home/jasper/dev/robocup_project/training-coding/'
    'models/roboheads-ssd_mobilenet_v1/frozen_inference_graph.pb'
)
PATH_TO_IMAGE = (
    '/home/jasper/dev/robocup_project/reference-images/full/'
    'converted-1m-front-left-full.png'
)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_image_into_numpy_array_cv(image):
    (im_width, im_height, _) = image.shape
    return np.array(image.data).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            tf.train.write_graph(sess.graph, '/tmp/notebook-model',
                                 'train.pbtxt')
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs
            }
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    graph = tf.get_default_graph()
                    tensor_dict[key] = graph.get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],
                                             [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'],
                                             [0])
                # Reframe is required to translate mask from box coordinates
                # to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0],
                                             tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0],
                                           [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                           [real_num_detection, -1, -1])
                detection_masks_reframed = \
                    utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0],
                        image.shape[1]
                    )
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name(
                'image_tensor:0')

            # Run inference
            output_dict = sess.run(
                tensor_dict,
                feed_dict={image_tensor: np.expand_dims(image, 0)}
            )

            # all outputs are float32 numpy arrays, so convert types as
            # appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0]
            )
            output_dict['detection_classes'] = output_dict[
                'detection_classes'
            ][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict[
                'detection_scores'
            ][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict[
                    'detection_masks'
                ][0]
    return output_dict


image = cv2.cvtColor(cv2.imread(PATH_TO_IMAGE), cv2.COLOR_BGR2RGB)
results = run_inference_for_single_image(image, detection_graph)
m_v = -1
m_i = 0
for i, v in enumerate(results['detection_scores']):
    if v > m_v:
        m_v = v
        m_i = i
print(repr({
    'box': results['detection_boxes'][m_i],
    'score': m_v
}))
