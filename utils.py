import tensorflow as tf
import os, sys
from glob import glob
import numpy as np



# fixme: 정녕 이 방법 뿐인가 ,,
def crop_to_multiple_bounding_box(batch_image, batch_mask_coord):

    image_shape = batch_image[0].get_shape().as_list()
    flatten_shape = np.cumprod(image_shape)[-1]
    _image = tf.reshape(batch_image, [-1, flatten_shape])
    _image = tf.cast(_image, tf.int32)

    # mask_args = tf.stack(batch_mask_coord, axis=1)

    crop_args = tf.concat([_image, batch_mask_coord], axis=1)

    def crop_and_resize(x):

        im, mh, mw, mhs, mws = tf.split(x, [flatten_shape, 1, 1, 1, 1], axis=0)
        im = tf.reshape(im, [-1]+image_shape)

        crop = tf.image.crop_to_bounding_box(im, mh, mw, mhs, mws)

        image_size = image_shape[1]
        reduced_shape = [int(image_size // 4), int(image_size // 4)]
        crop_resize = tf.image.resize_images(crop, reduced_shape)

        return crop_resize

    batch_cropped_image = tf.map_fn(lambda x: crop_and_resize(x), crop_args)

    return batch_cropped_image




    # batch_cropped_image = tf.map_fn(lambda x: crop_helper(x), crop_args)
    #
    # image_size = batch_image.get_shape().as_list()[1]
    # reduced_shape = [int(image_size // 4), int(image_size // 4)]
    # batch_cropped_image = tf.map_fn(lambda x: tf.image.resize_images(x, reduced_shape), batch_cropped_image)
    #
    # # batch_cropped_image = tf.stack(batch_cropped_image, axis=0)



    # batch_size = batch_image.get_shape().as_list()[0]
    # cropped_image = [tf.image.crop_to_bounding_box(
    #         image, mh, mw, mhs, mws) for (image, mh, mw, mhs, mws) in ((batch_image,) + batch_mask_coord)
    # ]
# def crop_to_multiple_bounding_box(batch_image, batch_mh, batch_mw, batch_mhs, batch_bws):
#     batch_size = batch_image.get_shape().as_list()[0]
#     image_size = batch_image.get_shape().as_list()[1]
#     cropped_images = []
#
#     cropped_images = [tf.image.crop_to_bounding_box(
#             image, mh, mw, mhs, ws) for (image, mh, mw, mhs, mws) in zip(batch_image, )
#     ]
#     for i in range(batch_size):
#         cropped_image = tf.image.crop_to_bounding_box(
#             batch_image[i], batch_mh[i], batch_mw[i], batch_mhs[i], batch_mws[i])
#
#         cropped_image = tf.image.resize_images(cropped_image, [int(image_size // 4), int(image_size // 4)])  # check
#         cropped_images.append(cropped_image)
#
#     batch_cropped_image = tf.stack(cropped_images, axis=0)
#
#     return batch_cropped_image


# def load_data_2(data_dir, image_size, batch_size, mode='train'):
#
#     if mode == 'train':
#         paths = os.path.join('Development/mk/cram/data', data_dir, 'train/*.jpg')
#     else:
#         paths = os.path.join('Development/mk/cram/data', data_dir, 'test/*.jpg')
#
#     data_count = len(glob(paths))
#     filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))
#
#     image_reader = tf.WholeFileReader()
#     _, image_file = image_reader.read(filename_queue)
#     image = tf.image.decode_jpeg(image_file, channels=3)
#
#     # Need to predefine the tensor shape
#     image = tf.image.resize_images(image, [image_size, image_size])
#     # image = tf.image.convert_image_dtype(image, dtype=tf.float32)  #
#
#     m_w = tf.random_uniform([1], 0, int(image_size/4*3), dtype=tf.int32)[0]
#     m_h = tf.random_uniform([1], 0, int(image_size/4*3), dtype=tf.int32)[0]
#     m_ws = tf.random_uniform([1], int(image_size//4), image_size-m_w, dtype=tf.int32)[0]
#     m_hs = tf.random_uniform([1], int(image_size//4), image_size-m_h, dtype=tf.int32)[0]
#
#     orig_image = image
#
#     part_image = tf.image.crop_to_bounding_box(orig_image, m_h, m_w, m_hs, m_ws)
#     part_image = tf.image.resize_images(part_image, [int(image_size//4), int(image_size//4)])  # check
#
#     # todo: tf.pad -- tensor cannot be modified with index
#     mask = tf.zeros_like(orig_image)
#     mask[m_h:m_h+m_hs, m_w:m_ws].assign(1.0)
#
#     masked_image = tf.multiply(orig_image, mask)
#
#     orig_image, masked_image, mask, part_image = tf.train.shuffle_batch(
#         [orig_image, masked_image, mask, part_image],
#         batch_size=batch_size,
#         capacity=batch_size * 4,
#         min_after_dequeue=batch_size
#         )
#
#     return orig_image, masked_image, mask, part_image, data_count

    # mask_coord = m_w, m_h, m_ws, m_hs
    # orig_image = image
    #
    # mask_coord, orig_image = tf.train.shuffle_batch(
    #     [mask_coord, orig_image],
    #     batch_size=batch_size,
    #     capacity=batch_size * 4,
    #     min_after_dequeue=batch_size
    #     )
    #
    # return mask_coord, orig_image, data_count


