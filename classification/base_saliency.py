import tensorflow as tf
import os, sys
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

slim = tf.contrib.slim

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


_s = 256

class dataloader(object):
    def __init__(self, data_dir, batch_size, data_format='jpg', mode='train'):
        self.batch_size = batch_size

        #         if mode == 'train':
        #             self.image_paths = os.path.join('../data', data_dir, 'train/*.' + data_format)
        #             self.mask_paths = os.path.join('../data', data_dir, 'train/*.' + data_format)

        self.image_paths = os.path.join('../data', data_dir, 'images/*.' + 'jpg')
        self.image_paths = sorted(glob(self.image_paths))
        self.mask_paths = os.path.join('../data', data_dir, 'gt_masks/*.' + 'png')
        self.mask_paths = sorted(glob(self.mask_paths))

        self.data_count = len(self.image_paths)
        self.num_batch = int(self.data_count / self.batch_size)
        self.pointer = 0

    def next_batch(self):
        self.pointer = (self.pointer + 1) % self.num_batch

        start_pos = self.pointer * self.batch_size
        batch_images = np.empty(shape=[self.batch_size, _s, _s, 3])

        for i, _i in zip(range(self.batch_size),
                         self.image_paths[start_pos:start_pos + self.batch_size]):
            # image = tf.image.decode_jpeg(i, channel=3)
            # image = tf.image.resize_images(image, size=[256, 256])
            image = Image.open(_i)
            image = image.resize(size=[_s, _s])
            batch_images[i] = np.array(image)


        batch_masks = np.empty(shape=[self.batch_size, _s, _s])
        for i, _m in zip(range(self.batch_size),
                          self.mask_paths[start_pos:start_pos + self.batch_size]):
            # mask = tf.image.decode_png(i, channel=1)
            # mask = tf.image.resize_images(mask, size=[256, 256])
            mask = Image.open(_m)
            mask = mask.resize(size=[_s, _s])
            batch_masks[i] = np.array(mask)


        return batch_images, batch_masks

# data load
data = dataloader('MSRA10K', 32)

# placeholder

x = tf.placeholder(tf.float32, shape=[None, _s, _s, 3], name='x')
y = tf.placeholder(tf.float32, shape=[None, _s, _s], name='y')

# model
# encoder-decoder
net = x  # input

n_filters = [16, 32, 64, 128, 256]
with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                    activation_fn=tf.nn.elu,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    normalizer_fn=slim.batch_norm,
                    ):
    for i in range(len(n_filters)):
        net = slim.conv2d(net, n_filters[i], 3, stride=2, padding='SAME',
                          scope='conv' + str(i))  # (?, 96, 136, 30)

    # decoder
    for i in range(len(n_filters)):
        net = slim.conv2d_transpose(net, n_filters[::-1][i], 3, stride=2, padding='SAME',
                                    scope='deconv' + str(i))

net = slim.conv2d(net, 1, (3, 3), padding='SAME', scope='reshape')  # (?, 96, 136, 30)

pred = tf.nn.sigmoid(net)
pred = tf.reshape(pred, shape=[-1, _s, _s])

# loss, optimizer
# l1_loss = tf.reduce_sum(tf.abs(tf.subtract(y, pred)))
l2_loss = tf.nn.l2_loss(tf.subtract(y, pred))
optim = tf.train.AdamOptimizer(0.0001, 0.9).minimize(l2_loss)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # for i in tqdm(range(1000)):
    for i in range(100000):
        batch_xs, batch_ys = data.next_batch()
        l, _ = sess.run([l2_loss, optim], feed_dict={x: batch_xs, y: batch_ys})
        if i % 50 == 0:
            print(i, l)

    print('image size: ', _s)
    saver.save(sess, 'base_saliency/' + str(_s) + "/model.ckpt")
