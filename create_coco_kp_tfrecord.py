import collections
import hashlib
import json
import logging
import multiprocessing
import os
import numpy as np
import cv2
import math
import tensorflow.compat.v1 as tf
import label_map_util
import tfrecord_util
from utils import _full_image_crop, _resize_image, _clip_detections, \
    gaussian_radius, draw_gaussian

with open('config.json', 'r') as f:
    configs = json.load(f)


def create_tf_example(
        image,
        image_dir,
        obj_detections=None,
        category_index=None):
    """
      Converts image and annotations to a tf.Example proto.

      Args:
        image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
          u'width', u'date_captured', u'flickr_url', u'id']
        image_dir: directory containing the image files.

        obj_detections: a dict containing bounding boxes and category of every object.
            See the _load_object_annotations function.
        category_index: a dict containing COCO category information keyed by the
          'id' field of each category.  See the label_map_util.create_category_index
          function.
      Returns:
        example: The converted tf.Example

      Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    filename = image['file_name']
    image_id = image['id']

    gaussian_bump = configs['gaussian_bump']
    gaussian_iou = configs['gaussian_iou']
    gaussian_rad = configs['gaussian_radius']
    num_classes = configs['num_classes']
    output_size_height = output_size_width = configs['output_size']
    input_size = configs['input_size']
    max_tag_len = len(obj_detections) + 1

    # read image
    full_path = os.path.join(image_dir, filename)
    image = cv2.imread(full_path)
    # cropping an image randomly
    detections = np.asarray(obj_detections)

    image, detections = _full_image_crop(image, detections)
    image, detections = _resize_image(image, detections, input_size)
    detections = _clip_detections(image, detections)
    width_ratio = output_size_width / input_size
    height_ratio = output_size_height / input_size

    key = hashlib.sha256(image).hexdigest()
    success, encoded_image = cv2.imencode('.jpg', image)
    encoded_image = encoded_image.tobytes()
    feature_dict = {
        'image/height':
            tfrecord_util.int64_feature(input_size),
        'image/width':
            tfrecord_util.int64_feature(input_size),
        'image/filename':
            tfrecord_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            tfrecord_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            tfrecord_util.bytes_feature(encoded_image),
        'image/format':
            tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
    }

    # focal loss
    tl_heatmaps = np.zeros((num_classes, output_size_height, output_size_width), dtype=np.float32)
    br_heatmaps = np.zeros((num_classes, output_size_height, output_size_width), dtype=np.float32)
    ct_heatmaps = np.zeros((num_classes, output_size_height, output_size_width), dtype=np.float32)
    # off loss [max_range, 2], max_range=128 in CenterNet
    tl_regrs = np.zeros((max_tag_len, 2), dtype=np.float32)
    br_regrs = np.zeros((max_tag_len, 2), dtype=np.float32)
    ct_regrs = np.zeros((max_tag_len, 2), dtype=np.float32)
    # push and pull loss
    tl_tags = np.zeros((max_tag_len), dtype=np.int64)
    br_tags = np.zeros((max_tag_len), dtype=np.int64)
    ct_tags = np.zeros((max_tag_len), dtype=np.int64)
    # ordered object in annotation [max_range]
    tag_masks = np.zeros((max_tag_len), dtype=np.uint8)
    category_names = []
    category_ids = []

    # ind: image_id, detection: bboxes, categories
    for ind, detection in enumerate(detections):
        category = int(detection[-1]) - 1
        category_ids.append(category)
        category_names.append(category_index[category]['name'].encode('utf8'))

        xtl, ytl = detection[0], detection[1]
        xbr, ybr = detection[2], detection[3]
        xct, yct = (xtl + xbr) / 2., (ytl + ybr) / 2.

        fxtl = (xtl * width_ratio)
        fytl = (ytl * height_ratio)
        fxbr = (xbr * width_ratio)
        fybr = (ybr * height_ratio)
        fxct = (xct * width_ratio)
        fyct = (yct * height_ratio)

        xtl = int(fxtl)
        ytl = int(fytl)
        xbr = int(fxbr)
        ybr = int(fybr)
        xct = int(fxct)
        yct = int(fyct)

        if gaussian_bump:
            width = detection[2] - detection[0]
            height = detection[3] - detection[1]

            width = math.ceil(width * width_ratio)
            height = math.ceil(height * height_ratio)

            if gaussian_rad == -1:
                radius = gaussian_radius((height, width), gaussian_iou)
                radius = max(0, int(radius))
            else:
                radius = gaussian_rad

            draw_gaussian(tl_heatmaps[category], [xtl, ytl], radius)
            draw_gaussian(br_heatmaps[category], [xbr, ybr], radius)
            draw_gaussian(ct_heatmaps[category], [xct, yct], radius, delte=5)

        else:
            tl_heatmaps[category, ytl, xtl] = 1
            br_heatmaps[category, ybr, xbr] = 1
            ct_heatmaps[category, yct, xct] = 1

        tl_regrs[ind, :] = [fxtl - xtl, fytl - ytl]
        br_regrs[ind, :] = [fxbr - xbr, fybr - ybr]
        ct_regrs[ind, :] = [fxct - xct, fyct - yct]
        tl_tags[ind] = ytl * output_size_width + xtl
        br_tags[ind] = ybr * output_size_width + xbr
        ct_tags[ind] = yct * output_size_width + xct
        tag_masks[:ind + 1] = 1

    tl_tags = tfrecord_util.serialize_array(tl_tags)
    br_tags = tfrecord_util.serialize_array(br_tags)
    ct_tags = tfrecord_util.serialize_array(ct_tags)
    tag_masks = tfrecord_util.serialize_array(tag_masks)
    tl_regrs = tfrecord_util.serialize_array(tl_regrs)
    br_regrs = tfrecord_util.serialize_array(br_regrs)
    ct_regrs = tfrecord_util.serialize_array(ct_regrs)
    tl_heatmaps = tfrecord_util.serialize_array(tl_heatmaps)
    br_heatmaps = tfrecord_util.serialize_array(br_heatmaps)
    ct_heatmaps = tfrecord_util.serialize_array(ct_heatmaps)

    feature_dict.update({
        'xs/tl_tags':
            tfrecord_util.bytes_feature(tl_tags),
        'xs/br_tags':
            tfrecord_util.bytes_feature(br_tags),
        'xs/ct_tags':
            tfrecord_util.bytes_feature(ct_tags),
        'xs/category_names':
            tfrecord_util.bytes_list_feature(category_names),
        'xs/category_ids':
            tfrecord_util.int64_list_feature(category_ids),
        'ys/tag_masks':
            tfrecord_util.bytes_feature(tag_masks),
        'ys/tl_regrs':
            tfrecord_util.bytes_feature(tl_regrs),
        'ys/br_regrs':
            tfrecord_util.bytes_feature(br_regrs),
        'ys/ct_regrs':
            tfrecord_util.bytes_feature(ct_regrs),
        'ys/tl_heatmaps':
            tfrecord_util.bytes_feature(tl_heatmaps),
        'ys/br_heatmaps':
            tfrecord_util.bytes_feature(br_heatmaps),
        'ys/ct_heatmaps':
            tfrecord_util.bytes_feature(ct_heatmaps)
    })

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example


def _pool_create_tf_example(args):
    return create_tf_example(*args)


def _load_object_annotations(object_annotations_file):
    """Loads object annotation JSON file."""
    with tf.gfile.GFile(object_annotations_file, 'r') as fid:
        obj_annotations = json.load(fid)

    category_index = label_map_util.create_category_index(
        obj_annotations['categories'])

    img_to_obj_detection = collections.defaultdict(list)
    logging.info('Building bounding box index.')
    for annotation in obj_annotations['annotations']:
        image_id = annotation['image_id']
        # make coordinate top-left and bottom-right
        bboxes = np.array(annotation['bbox'], dtype=float)
        bboxes = bboxes + np.hstack((np.zeros(2, dtype=float), bboxes[0:2]))
        categories = annotation['category_id']
        if bboxes.size == 0 or categories == 0:
            img_to_obj_detection[image_id].append(np.zeros((0, 5), dtype=np.float32))
        else:
            img_to_obj_detection[image_id].append(np.hstack((bboxes, categories)))

    return category_index, img_to_obj_detection


def _load_images_info(images_info_file):
    with tf.gfile.GFile(images_info_file, 'r') as fid:
        info_dict = json.load(fid)
    return info_dict['images']


def _create_tf_record_from_coco_annotations(
        images_info_file,
        image_dir,
        output_path,
        num_shards):
    """
      Loads COCO annotation json files and converts to tf.Record format.

      Args:
        images_info_file: JSON file containing bounding box annotations.
        image_dir: Directory containing the image files.
        output_path: Path to output tf.Record file.
        num_shards: Number of output files to create.
    """

    logging.info('writing to output path: %s', output_path)
    writers = [
        tf.python_io.TFRecordWriter(
            output_path + '-%05d-of-%05d.tfrecord' % (i, num_shards))
        for i in range(num_shards)
    ]

    category_index = None
    if images_info_file:
        category_index, img_to_obj_detection = (
            _load_object_annotations(images_info_file))

    def _get_object_detection(image_id):
        if img_to_obj_detection:
            return img_to_obj_detection[image_id]
        else:
            return None

    images = _load_images_info(images_info_file)
    pool = multiprocessing.Pool()
    for idx, (_, tf_example) in enumerate(
            pool.imap(_pool_create_tf_example,
                      [(image, image_dir, _get_object_detection(image['id']), category_index) for image in images])):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(images))

        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()

    for writer in writers:
        writer.close()


def main(_):
    directory = os.path.dirname(configs['output_file_prefix'])
    if not tf.gfile.IsDirectory(directory):
        tf.gfile.MakeDirs(directory)

    _create_tf_record_from_coco_annotations(
        configs['object_annotations_file'],
        configs['image_dir'],
        configs['output_file_prefix'],
        configs['num_shards']
    )


if __name__ == '__main__':
    tf.app.run(main)
