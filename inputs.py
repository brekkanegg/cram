import tensorflow as tf
import os, sys
from glob import glob
import itertools

from scipy.io import loadmat
import numpy as np

"""
issue: 이미지 스케일: 0-1
"""



def inputs_svhn(data_dir, image_size, batch_size, num_threads, mode='train', central_crop=True):
    data = loadmat('data/svhn/{}_32x32.mat'.format(mode))
    # data = loadmat('Development/mk/cram/data/svhn/{}_32x32.mat'.format(mode))

    data = np.einsum('hwcn->nhwc', data['X'])

    data_count = len(data)
    image_size = data[0].shape[0]  # 32

    dataset = tf.data.Dataset.from_tensor_slices(data)

    def image_mask_central_crop(image, image_size, crop_ratio=0.25):
        # image = tf.image.decode_jpeg(image, channels=3)
        # Need to predefine the tensor shape, check resize method
        # image = tf.image.resize_images(image, [image_size, image_size],
        #                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  #

        mask = tf.ones_like(image, dtype=tf.float32)
        mask = tf.image.central_crop(mask, crop_ratio)
        mask_size = int(image_size * crop_ratio)
        mx = int((image_size - mask_size) / 2)
        mask = tf.image.pad_to_bounding_box(mask, mx, mx, image_size, image_size)

        orig_image = image
        part_image = tf.image.central_crop(image, crop_ratio)
        part_image = tf.image.resize_images(part_image, [int(image_size // 4), int(image_size // 4)],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image_mask_dict = {
            'orig_image': orig_image,
            'part_image': part_image,
            'mask': mask,
            'mask_coord': tf.constant(0)  # not necessary for central crop
        }

        return image_mask_dict

    dataset = dataset.map(lambda x: image_mask_central_crop(x, image_size), num_threads)
    dataset = dataset.batch(batch_size)
    dataset.repeat()

    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)

    return next_element, init_op, data_count



def inputs_big(data_dir, image_size, batch_size, num_threads, mode='train', central_crop=True):

    data_format = 'jpg'

    # fixme: 상대경로
    if mode == 'train':
        paths = os.path.join('data', data_dir, 'train/*.' + data_format)
        # paths = os.path.join('Development/mk/cram/data', data_dir, 'train/*.' + data_format)

        if data_dir == 'places365':  # use test instead
            paths = os.path.join('data', data_dir, 'train/*/*/*.' + data_format)
            # paths = os.path.join('Development/mk/cram/data', data_dir, 'train/*/*/*.' + data_format)

    else:
        paths = os.path.join('Development/mk/cram/data', data_dir, 'test/*.' + data_format)
        # paths = os.path.join('Development/mk/cram/data', data_dir, 'test/*.' + data_format)

    data_count = len(glob(paths))

    filenames = tf.train.match_filenames_once(paths)

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if central_crop:
        dataset = dataset.map(lambda x: image_mask_central_crop(x, image_size), num_threads)
    else:  # random_crop
        dataset = dataset.map(lambda x: image_mask_random_crop(x, image_size), num_threads)
    dataset = dataset.batch(batch_size)
    dataset.repeat()

    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)

    return next_element, init_op, data_count


# fixme: mask_coord 가 배치로 안 뜸

def image_mask_central_crop(image, image_size, crop_ratio=0.25):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    # Need to predefine the tensor shape, check resize method
    image = tf.image.resize_images(image, [image_size, image_size],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  #

    mask = tf.ones_like(image, dtype=tf.float32)
    mask = tf.image.central_crop(mask, crop_ratio)
    mask_size = int(image_size * crop_ratio)
    mx = int((image_size - mask_size) / 2)
    mask = tf.image.pad_to_bounding_box(mask, mx, mx, image_size, image_size)

    orig_image = image
    part_image = tf.image.central_crop(image, crop_ratio)
    part_image = tf.image.resize_images(part_image, [int(image_size // 4), int(image_size // 4)],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_mask_dict = {
        'orig_image': orig_image,
        'part_image': part_image,
        'mask': mask,
        'mask_coord': tf.constant(0)  # not necessary for central crop
    }

    return image_mask_dict


# fixme: mask_coord 가 배치로 안 뜸
def image_mask_random_crop(image, image_size):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    # Need to predefine the tensor shape
    image = tf.image.resize_images(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  #
    inputs.py
    m_w = tf.random_uniform([1], 0, int(image_size/4*3), dtype=tf.int32)[0]
    m_h = tf.random_uniform([1], 0, int(image_size/4*3), dtype=tf.int32)[0]
    m_ws = tf.random_uniform([1], int(image_size//4), image_size-m_w, dtype=tf.int32)[0]
    m_hs = tf.random_uniform([1], int(image_size//4), image_size-m_h, dtype=tf.int32)[0]
    mask_coord = tf.stack([m_h, m_w, m_hs, m_ws], axis=0)

    part_image = tf.image.crop_to_bounding_box(image, m_h, m_w, m_hs, m_ws)
    mask = tf.image.pad_to_bounding_box(part_image, m_h, m_w, image_size, image_size)
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)

    orig_image = image
    part_image = tf.image.resize_images(part_image, [int(image_size//4), int(image_size//4)])

    image_mask_dict = {
        'orig_image': orig_image,
        'part_image': part_image,
        'mask': mask,
        'mask_coord': mask_coord
    }

    return image_mask_dict

