#!/usr/bin/env python3
'''
Convert some input data (generated by `coding.py`) to a TFRecord.

Run as

```
./to-tfrecord --data data.json --data vito.json --test data/test.record \
    --train data/train.record
```
'''

import json
from PIL import Image
import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('train', None, 'Path to training TFRecord')
flags.DEFINE_string('test', None, 'Path to test TFRecord')
flags.DEFINE_multi_string('data', None, 'Path to input data')
flags.mark_flag_as_required('train')
flags.mark_flag_as_required('test')
flags.mark_flag_as_required('data')
FLAGS = flags.FLAGS


def minmax(a, b):
    '''Returns a and b in sorted order'''
    if a > b:
        return b, a
    return a, b


def bound(val, lower=0, upper=1):
    '''Return val if it is in [lower..upper], else the respective value.'''
    return max(lower, min(upper, val))


def normalize(head, width, height):
    '''Normalizes head coordinates'''
    xmin, xmax = minmax(head[0][0], head[1][0])
    xmin = bound(xmin / width)
    xmax = bound(xmax / width)
    ymin, ymax = minmax(head[0][1], head[1][1])
    ymin = bound(ymin / height)
    ymax = bound(ymax / height)
    return xmin, xmax, ymin, ymax


def tf_example(path, heads):
    '''Creates a TF Example from the raw data.'''
    width, height, img, img_format = read_image(path)
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for head in heads:
        xmin, xmax, ymin, ymax = normalize(head, width, height)
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
    classes_txt = [b'Head'] * len(heads)
    classes = [1] * len(heads)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/width': dataset_util.int64_feature(width),
        'image/height': dataset_util.int64_feature(height),
        'image/filename': dataset_util.bytes_feature(path.encode()),
        'image/source_id': dataset_util.bytes_feature(path.encode()),
        'image/encoded': dataset_util.bytes_feature(img),
        'image/format': dataset_util.bytes_feature(img_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_txt),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))


def read_image(path):
    '''Reads an image from disk an returns the image along with some metadata.'''
    with open(path, 'rb') as fh:
        data = fh.read()
        img = Image.open(path)
        if img.format == 'JPEG':
            fmt = b'jpg'
        elif img.format == 'PNG':
            fmt = b'png'
        else:
            raise RuntimeError('Unknown format: {}'.format(img.format))
        width, height = img.size
        return width, height, data, fmt


def read_data(paths):
    '''Reads the input data and splits it into training and test data'''
    data = {}
    for path in paths:
        with open(path) as fh:
            data = {**data, **json.load(fh)}

    train, test = {}, {}
    for path, heads in data.items():
        # 10% test data, 90% training data. By using hash, the results are
        # reproducible.
        if hash(path) % 10 > 0:
            train[path] = heads
        else:
            test[path] = heads

    return train, test


def convert(data, output):
    writer = tf.python_io.TFRecordWriter(output)

    for path, heads in data.items():
        print('Reading {}...'.format(path))
        example = tf_example(path, heads)
        writer.write(example.SerializeToString())

    writer.close()


def main(_):
    '''Read, convert and write the data.'''
    train, test = read_data(FLAGS.data)

    convert(train, FLAGS.train)
    convert(test, FLAGS.test)
    print('Converted {} training and {} testing examples'.format(
        len(train), len(test)))


if __name__ == '__main__':
    tf.app.run()
