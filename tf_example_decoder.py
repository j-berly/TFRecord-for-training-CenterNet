# https://github.com/google/automl/blob/master/efficientdet/object_detection/tf_example_decoder.py
import tensorflow.compat.v1 as tf


def decode_image(parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
    return image


def get_source_id_from_encoded_image(parsed_tensors):
    return tf.strings.as_string(
        tf.strings.to_hash_bucket_fast(parsed_tensors['image/encoded'], 2 ** 63 - 1))


def decode(serialized_example):
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string),
      'image/source_id': tf.FixedLenFeature((), tf.string, ''),
      'image/height': tf.FixedLenFeature((), tf.int64, -1),
      'image/width': tf.FixedLenFeature((), tf.int64, -1),
      'xs/tl_tags': tf.FixedLenFeature((), tf.string),
      'xs/br_tags': tf.FixedLenFeature((), tf.string),
      'xs/ct_tags': tf.FixedLenFeature((), tf.string),
      'xs/category_ids': tf.VarLenFeature(tf.int64),
      'ys/tag_masks': tf.FixedLenFeature((), tf.string),
      'ys/tl_regrs': tf.FixedLenFeature((), tf.string),
      'ys/br_regrs': tf.FixedLenFeature((), tf.string),
      'ys/ct_regrs': tf.FixedLenFeature((), tf.string),
      'ys/tl_heatmaps': tf.FixedLenFeature((), tf.string),
      'ys/br_heatmaps': tf.FixedLenFeature((), tf.string),
      'ys/ct_heatmaps': tf.FixedLenFeature((), tf.string),
    }
    example_message = tf.io.parse_single_example(serialized_example, keys_to_features)

    image = decode_image(example_message)
    decode_image_shape = tf.logical_or(
        tf.equal(example_message['image/height'], -1),
        tf.equal(example_message['image/width'], -1))
    image_shape = tf.cast(tf.shape(image), dtype=tf.int64)

    example_message['image/height'] = tf.where(decode_image_shape,
                                               image_shape[0],
                                               example_message['image/height'])
    example_message['image/width'] = tf.where(decode_image_shape, image_shape[1],
                                              example_message['image/width'])

    source_id = tf.cond(
            tf.greater(tf.strings.length(example_message['image/source_id']),
                       0), lambda: example_message['image/source_id'],
            lambda: get_source_id_from_encoded_image(example_message))

    decoded_tensors = {
        'image': image,
        'source_id': source_id,
        'height': example_message['image/height'],
        'width': example_message['image/width'],
        'tl_tags': tf.io.parse_tensor(example_message['xs/tl_tags'], out_type=tf.int64),
        'br_tags': tf.io.parse_tensor(example_message['xs/br_tags'], out_type=tf.int64),
        'ct_tags': tf.io.parse_tensor(example_message['xs/ct_tags'], out_type=tf.int64),
        'category_ids': example_message['xs/category_ids'],
        'tl_heatmaps': tf.io.parse_tensor(example_message['ys/tl_heatmaps'], out_type=tf.float32),
        'br_heatmaps': tf.io.parse_tensor(example_message['ys/br_heatmaps'], out_type=tf.float32),
        'ct_heatmaps': tf.io.parse_tensor(example_message['ys/ct_heatmaps'], out_type=tf.float32),
        'tl_regrs': tf.io.parse_tensor(example_message['ys/tl_regrs'], out_type=tf.float32),
        'br_regrs': tf.io.parse_tensor(example_message['ys/br_regrs'], out_type=tf.float32),
        'ct_regrs': tf.io.parse_tensor(example_message['ys/ct_regrs'], out_type=tf.float32),
        'tag_masks': tf.io.parse_tensor(example_message['ys/tag_masks'], out_type=tf.uint8),
    }
    return decoded_tensors


tfr_dataset = tf.data.TFRecordDataset('kp/train-00000-of-00001.tfrecord')

dataset = tfr_dataset.map(decode, num_parallel_calls=64)

for instance in dataset:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    check if the image decode sucessfully
    arr_ = np.squeeze(instance['image']) 
    plt.imshow(arr_)
    plt.show()
    """
    print(instance) # print parsed example messages with restored arrays
