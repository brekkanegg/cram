"""
use binary cross entropy?
"""
# todo: use binary cross entropy?
# fixme: learning rate 줄이는 방법

import tensorflow as tf
import os, sys
import numpy as np
from glob import glob
from PIL import Image
# from tqdm import tqdm
import time, datetime
import random

slim = tf.contrib.slim



st = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d%H%M')


flags = tf.app.flags
flags.DEFINE_integer("image_size", 64, "The size of images [32]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [16]")
flags.DEFINE_integer("epoch", 200, "The num of epoch")

flags.DEFINE_integer("iteration", 40000, "The num of iteration")
flags.DEFINE_float("learning_rate", 0.01, "The learning rate [0.01]")

flags.DEFINE_bool("override", False, "override")


flags.DEFINE_string("gpu", "1", "# of gpu to use"),
FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

_s = FLAGS.image_size
_bs = FLAGS.batch_size
_it = FLAGS.iteration
_ep = FLAGS.epoch
_lr = FLAGS.learning_rate

class dataloader(object):
    def __init__(self, data_dir, batch_size, data_format='jpg', mode='train'):
        self.batch_size = batch_size

        self.image_paths = os.path.join('data', data_dir, 'images/*.' + data_format)
        self.image_paths = sorted(glob(self.image_paths))
        self.mask_paths = os.path.join('data', data_dir, 'gt_masks/*.' + 'png')
        self.mask_paths = sorted(glob(self.mask_paths))

        cut = int(len(self.image_paths)*0.8)  # train, val ratio: 0.8
        if mode == 'train':
            self.image_paths = self.image_paths[:cut]
            self.mask_paths = self.mask_paths[:cut]
        elif mode == 'val':
            self.image_paths = self.image_paths[cut:]
            self.mask_paths = self.mask_paths[cut:]


        self.data_count = len(self.image_paths)
        self.num_batch = int(self.data_count / self.batch_size)
        self.pointer = 0

    def next_batch(self):
        self.pointer = (self.pointer + 1) % self.num_batch

        start_pos = self.pointer * self.batch_size

        batch_images = np.empty(shape=[self.batch_size, _s, _s, 3])
        for i, im in zip(range(self.batch_size), self.image_paths[start_pos:start_pos + self.batch_size]):
            # image = tf.image.decode_jpeg(i, channel=3)
            # image = tf.image.resize_images(image, size=[256, 256])
            image = Image.open(im).convert("RGB")
            image = image.resize(size=[_s, _s])
            batch_images[i] = np.array(image)

        batch_masks = np.empty(shape=[self.batch_size, _s, _s])
        for i, m in zip(range(self.batch_size), self.mask_paths[start_pos:start_pos + self.batch_size]):
            # mask = tf.image.decode_png(i, channel=1)
            # mask = tf.image.resize_images(mask, size=[256, 256])
            mask = Image.open(m).convert("1")
            mask = mask.resize(size=[_s, _s])
            batch_masks[i] = np.array(mask)

        return batch_images, batch_masks

    def shuffle(self):
        combined = list(zip(self.image_paths, self.mask_paths))
        random.shuffle(combined)
        self.image_paths, self.mask_paths = zip(*combined)

# data load
train_data = dataloader('MSRA10K', _bs)
val_data = dataloader('MSRA10K', _bs)

# data = dataloader('HKU-IS', 16, data_format='png')



# placeholder


x = tf.placeholder(tf.float32, shape=[None, _s, _s, 3], name='x')
y = tf.placeholder(tf.float32, shape=[None, _s, _s], name='y')
is_training = tf.placeholder(tf.bool, name='is_training')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')


# model
# encoder-decoder
net = x  # input

# n_filters = [64, 128, 256, 512, 512]
n_filters = [16, 32, 64, 128, 128]

if _s == 224:  # vgg16
    n_filters = [16, 32, 64, 128, 128]
elif _s == 32:  # vgg16
    n_filters = [16, 32, 64]


with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(0.001)):
    with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                        activation_fn=tf.nn.elu, updates_collections=None, is_training=is_training):

        # encoder
        print('encoder')
        for i in range(len(n_filters)):
            net = slim.conv2d(net, n_filters[i], 3, stride=1, padding='SAME', scope='conv1_' + str(i))
            net = slim.batch_norm(net, scope='e_bn1_' + str(i))
            print(net.shape)
            net = slim.conv2d(net, n_filters[i], 3, stride=2, padding='SAME', scope='conv2_' + str(i))
            net = slim.batch_norm(net, scope='e_bn2_' + str(i))
            print(net.shape)

        # fc

        # todo:  use fc?????
        # net = slim.conv2d(net, n_filters[-1]*2, 3, stride=8, padding='VALID', scope='fc1')
        # print(net.shape)

        print('decoder')
        net = slim.conv2d(net, n_filters[-1]*2, 1, stride=1, padding='SAME', scope='fc2')
        net = slim.batch_norm(net, scope='fc_bn')
        print(net.shape)
        # net = slim.conv2d_transpose(net, n_filters[-1], 3, stride=8, padding='SAME', scope='deconv1_' + '-1')
        # print(net.shape)

        # decoder
        # print('decoder')
        for i in range(len(n_filters)):
            net = slim.conv2d_transpose(net, n_filters[::-1][i], 3, stride=2, padding='SAME', scope='deconv1_' + str(i))
            net = slim.batch_norm(net, scope='d_bn1_' + str(i))
            print(net.shape)
            net = slim.conv2d_transpose(net, n_filters[::-1][i], 3, stride=1, padding='SAME', scope='deconv2_' + str(i))
            net = slim.batch_norm(net, scope='d_bn2_' + str(i))
            print(net.shape)

        # reshape
        print('reshape')
        net = slim.conv2d_transpose(net, 4, 1, padding='SAME', scope='reshape1')
        net = slim.batch_norm(net, scope='r_bn')
        print(net.shape)

        # last layer
        net = slim.conv2d_transpose(net, 1, 1, padding='SAME', scope='reshape2')
        print(net.shape)

        pred1 = tf.nn.sigmoid(net)
        pred = tf.reshape(pred1, shape=[-1, _s, _s])

# loss, optimizer
# print(slim.flatten(pred).shape, slim.flatten(y).shape)
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=slim.flatten(pred), labels=slim.flatten(y)))
loss = tf.reduce_mean(tf.abs(tf.subtract(y, pred)))  # l1


optimizer = tf.train.AdamOptimizer(_lr, 0.9)
# apply gradient clipping
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)

gradients = tf.gradients(loss, tf.trainable_variables())

# summary
all_summary = tf.summary.merge([tf.summary.histogram('grad' + str(i), tf.norm(capped_gvs[i]))
                                for i in range(len(capped_gvs))] +
                               [tf.summary.image("image", x, max_outputs=4)] +
                               [tf.summary.image("pred_saliency", pred1, max_outputs=4)] +
                               [tf.summary.image("true_saliency", tf.reshape(y, shape=[-1, _s, _s, 1]), max_outputs=4)])


#
checkpoint_dir = os.path.join('base_saliency', 'checkpoint', str(_s))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

summary_dir = os.path.join('base_saliency', 'summary', str(_s))
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:


    sess.run(tf.local_variables_initializer())

    # Try Loading Checkpoint
    print('Checkpoint: ', checkpoint_dir)
    print(" [*] Reading Checkpoint...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path and not FLAGS.override:  # retrain
        saver = tf.train.Saver(max_to_keep=5)
        t1 = time.time()
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restoring Time: ', time.time() - t1)

        # graph = tf.get_default_graph()
        # todo: global step
        # tf.train.get_or_create_global_step(graph)

        print("""
======
An existing model was found in the checkpoint directory.
Loading...
======

            """)
    else:
        print("""
======
An existing model was not found in the checkpoint directory.
Initializing a new one...
======
            """)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)

    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # for i in tqdm(range(1000)):
    print('Train Start!')
    start_time = time.time()

    min_val_loss = np.inf
    counter = 0
    stop_stack = 0

    batch_idxs = int(train_data.data_count // _bs)
    for e in range(_ep):
        if np.mod(e, 10) == 1 and e != 1:
            _lr *= 0.3
            print('learning rate decreased')

        train_data.shuffle()  # shuffle
        for i in range(0, batch_idxs):
            batch_xs, batch_ys = train_data.next_batch()
            l, s, _, = sess.run([loss, all_summary, train_op],
                                feed_dict={x: batch_xs, y: batch_ys, learning_rate: _lr, is_training: True})
            counter += 1
            # l, _ = sess.run([loss, optim], feed_dict={x: batch_xs, y: batch_ys})

            if counter % 50 == 0:
                print('epoch {} iteration: {} loss: {:.4f} time: {:.4f} '
                      .format(e, counter, l, time.time()-start_time))

                summary_writer.add_summary(s, i)

            # validation step
            if counter % 500 == 0:
                val_losses = []
                tot_batch_num = int(val_data.data_count / _bs)  # 16 is batch_size
                for idx in range(tot_batch_num):
                    batch_xs, batch_ys = val_data.next_batch()
                    vl = sess.run([loss], feed_dict={x: batch_xs, y: batch_ys, is_training: False})
                    val_losses.append(vl)

                val_loss = np.mean(val_losses)

                print("Recent Train Loss: {:.4f}, Validation Loss: {:.4f}".format(l, val_loss))
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    saver.save(sess, checkpoint_dir + '/saliency.ckpt')
                    print('Model saved at: {}/saliency.ckpt'.format(checkpoint_dir))
                else:
                    if l < min_val_loss * 0.7: ## overfitting
                        sys.exit(
                            "Stop Training! Iteration: {} Time Spent: {:.4f}".format(counter, time.time() - start_time))

                    # stop_stack += 1
                    # print('Stop Stack: {}/100, Iterations: {}'.format(stop_stack, counter))
                    # if stop_stack == 10:
                    #     sys.exit("Stop Training! Time Spent: {}".format(time.time()-start_time))

    print('Training finished, Time spent: {:.4f}'.format(time.time()-start_time))