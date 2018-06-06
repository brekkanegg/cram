import tensorflow as tf
import os
import numpy as np
from glob import glob
from PIL import Image
import random
from tensorflow.python.keras.utils import to_categorical

slim = tf.contrib.slim

"""
issue: 이미지 스케일: 0-1

feed_dict로 변경
"""

class base_saliency_model():

    def __init__(self, image_size, reuse):
        self.image_size = image_size
        self.checkpoint_dir = os.path.join('base_saliency', 'checkpoint', str(image_size))
        self.summary_dir = os.path.join('base_saliency', 'summary', str(image_size))

        self.build_model(reuse)


    def build_model(self, reuse):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)  # do not close session

        self.sess.run(tf.local_variables_initializer())
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3], name='x')

        # encoder-decoder
        net = self.x  # input

        if self.image_size == 224:  # vgg16
            n_filters = [16, 32, 64, 128, 128]
        elif self.image_size == 32:  # vgg16
            n_filters = [16, 32, 64]

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.001),
                            reuse=reuse):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.elu, updates_collections=None,
                                is_training=False, reuse=reuse):  #is_training: false

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
                # print(net.shape)5

                print('decoder')
                net = slim.conv2d(net, n_filters[-1] * 2, 1, stride=1, padding='SAME', scope='fc2')
                net = slim.batch_norm(net, scope='fc_bn')
                print(net.shape)
                # net = slim.conv2d_transpose(net, n_filters[-1], 3, stride=8, padding='SAME', scope='deconv1_' + '-1')
                # print(net.shape)

                # decoder
                # print('decoder')
                for i in range(len(n_filters)):
                    net = slim.conv2d_transpose(net, n_filters[::-1][i], 3, stride=2, padding='SAME',
                                                scope='deconv1_' + str(i))
                    net = slim.batch_norm(net, scope='d_bn1_' + str(i))
                    print(net.shape)
                    net = slim.conv2d_transpose(net, n_filters[::-1][i], 3, stride=1, padding='SAME',
                                                scope='deconv2_' + str(i))
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
                pred = tf.reshape(pred1, shape=[-1, self.image_size, self.image_size])

        # n_filters = [16, 32, 64, 128, 128]
        #
        # with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
        #                     activation_fn=tf.nn.relu,
        #                     weights_initializer=tf.contrib.layers.xavier_initializer(),
        #                     normalizer_fn=slim.batch_norm,
        #                     weights_regularizer=slim.l2_regularizer(0.001),
        #                     reuse=reuse):
        #     # encoder
        #     for i in range(len(n_filters)):
        #         net = slim.conv2d(net, n_filters[i], 3, stride=1, padding='SAME', scope='conv1_' + str(i))
        #         net = slim.conv2d(net, n_filters[i], 3, stride=2, padding='SAME', scope='conv2_' + str(i))
        #
        #     net = slim.conv2d(net, n_filters[-1] * 2, 1, stride=1, padding='SAME', scope='fc2')
        #
        #     # decoder
        #     for i in range(len(n_filters)):
        #         net = slim.conv2d_transpose(net, n_filters[::-1][i], 3, stride=2, padding='SAME',
        #                                     scope='deconv1_' + str(i))
        #         net = slim.conv2d_transpose(net, n_filters[::-1][i], 3, stride=1, padding='SAME',
        #                                     scope='deconv2_' + str(i))
        #
        #     # reshape
        #     net = slim.conv2d_transpose(net, 4, 1, padding='SAME', scope='reshape1')
        #     net = slim.conv2d_transpose(net, 1, 1, padding='SAME', activation_fn=None, scope='reshape2')
        #
        #     pred1 = tf.nn.sigmoid(net)
        #     pred = tf.reshape(pred1, shape=[-1, self.image_size, self.image_size])

            self.pred = pred1  # shape should be rank 4

        # load model
        print('\nCheckpoint of base saliency model: ', self.checkpoint_dir)
        print(" [*] Reading Checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        saver = tf.train.Saver(max_to_keep=4)
        saver.restore(self.sess, ckpt.model_checkpoint_path)


    def get_saliency(self, images):

        rough_saliency = self.sess.run([self.pred], feed_dict={self.x: images})
        rough_saliency = np.reshape(rough_saliency[0],
                                    [len(images), self.image_size, self.image_size, 1])
                                    # [images.shape[0], self.image_size, self.image_size, 1])

        return rough_saliency



################

class dataloader_cifar10(object):
    def __init__(self, batch_size, saliency=False, mode='train', reuse=False, sep=False, x255=False):
        self.saliency = saliency
        self.mode = mode
        self.image_size = 32
        self.class_num = 10
        self.sep = sep
        self.x255 = x255

        from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
        # if mode == 'train' or mode == 'control':
        #     (x, y), (_, _) = load_data()
        #     cut = int(len(x) * 0.8)
        #     x, y = x[:cut], y[:cut]
        #
        # elif mode == 'val':
        #     (x, y), (_, _) = load_data()
        #     cut = int(len(x) * 0.8)
        #     x, y = x[cut:], y[cut:]

        if mode == 'train': #or mode == 'control':
            (x, y), (_, _) = load_data()

        elif mode == 'val':
            (_, _), (x, y) = load_data()

        else:  # test, control
            (_, _), (x, y) = load_data()


        self.x = x
        y = y[:, 0]
        self.y = y

        # y_one_hot = to_categorical(y, num_classes=self.class_num)
        # self.y = y_one_hot

        if saliency:
            self.saliency_model = base_saliency_model(self.image_size, reuse=reuse)

        self.batch_size = batch_size
        self.data_count = x.shape[0]
        self.num_batch = int(self.data_count / self.batch_size)
        self.pointer = 0



    def next_batch(self):
        self.pointer = (self.pointer + 1) % self.num_batch

        start_pos = self.pointer * self.batch_size

        batch_images = self.x[start_pos:start_pos + self.batch_size]
        batch_labels = self.y[start_pos:start_pos + self.batch_size]
        if self.saliency:
            batch_saliencies = self.saliency_model.get_saliency(batch_images)
            if x255:
                batch_saliencies *= 255
            batch_images = np.concatenate([batch_images, batch_saliencies], axis=3)


        return batch_images, batch_labels

    def shuffle(self):
        combined = list(zip(self.x, self.y))
        random.shuffle(combined)
        self.x, self.y = zip(*combined)

###########

class dataloader_cub200(object):
    def __init__(self, batch_size, saliency=False, mode='train', reuse=False, sep=False, x255=False):
        self.saliency = saliency
        self.mode = mode
        self.image_size = 224
        self.class_num = 200
        self.sep = sep
        self.x255 = x255

        with open('data/CUB_200_2011/images.txt', 'r') as f:
            img_dir = f.read().split('\n')
            img_dir.remove('')
        with open('data/CUB_200_2011/train_test_split.txt', 'r') as f:
            train_test = f.read().split('\n')
            train_test.remove('')
        with open('data/CUB_200_2011/image_class_labels.txt', 'r') as f:
            img_class = f.read().split('\n')
            img_class.remove('')

        import pandas as pd
        df = pd.DataFrame()
        df['img_dir'] = ['data/CUB_200_2011/images/'+i.split(' ')[1] for i in img_dir]
        df['is_train'] = [i.split(' ')[1] for i in train_test]
        df['class'] = [i.split(' ')[1] for i in img_class]
        df['ann'] = ['data/CUB_200_2011/segmentations/' + i.split(' ')[1][:-3] + 'png' for i in img_dir]

        train, test = df[df['is_train'] == '1'], df[df['is_train'] == '0']

        if mode == 'train' or mode == 'gt':
            x, y = np.array(train['img_dir']), np.array(train['class']).astype('int') - 1

        elif mode == 'val':
            x, y = np.array(test['img_dir']), np.array(test['class']).astype('int') - 1


        else:  # test, control
            pass

        self.x = x
        self.y = y

        if mode == 'gt':
            self.s = np.array(train['ann'])
        elif saliency:
            self.saliency_model = base_saliency_model(self.image_size, reuse=reuse)

        self.batch_size = batch_size
        self.data_count = len(x)
        self.num_batch = int(self.data_count / self.batch_size)
        self.pointer = 0


    # fixme:
    def next_batch(self):
        self.pointer = (self.pointer + 1) % self.num_batch

        start_pos = self.pointer * self.batch_size

        batch_images_dir = self.x[start_pos:start_pos + self.batch_size]

        def to_rgb2(im):
            w, h = im.shape
            ret = np.empty((w, h, 3), dtype=np.uint8)
            ret[:, :, :] = im[:, :, np.newaxis]
            return ret

        temp_bi = [np.array(Image.open(_d).resize([self.image_size, self.image_size]))
                   if len(np.array(Image.open(_d)).shape) is 3
                   else to_rgb2(np.array(Image.open(_d).resize([self.image_size, self.image_size])))
                   for _d in batch_images_dir]
        batch_images = np.array(temp_bi)

        batch_labels = self.y[start_pos:start_pos + self.batch_size]

        if self.mode == 'gt':
            batch_saliencies_dir = self.s[start_pos:start_pos + self.batch_size]
            temp_bs = [np.array(Image.open(_d).resize([self.image_size, self.image_size]))
                       for _d in batch_saliencies_dir]
            #fixeme:

            _temp_bs = [np.reshape(b, [self.image_size, self.image_size, 1])
                        for b in temp_bs]
            batch_saliencies = np.array(_temp_bs)
            if self.x255:
                batch_saliencies *= 255
            batch_images = np.concatenate([batch_images, batch_saliencies], axis=3)

        elif self.saliency:
            batch_saliencies = self.saliency_model.get_saliency(batch_images)
            if self.x255:
                batch_saliencies *= 255
            batch_images = np.concatenate([batch_images, batch_saliencies], axis=3)

        return batch_images, batch_labels

    def shuffle(self):
        if self.mode == 'gt':
            combined = list(zip(self.x, self.y, self.s))
            random.shuffle(combined)
            self.x, self.y, self.s = zip(*combined)
        else:
            combined = list(zip(self.x, self.y))
            random.shuffle(combined)
            self.x, self.y = zip(*combined)

# outdated
# class dataloader_cub200(object):
#     def __init__(self, batch_size, saliency=False, mode='train'):
#         self.saliency = saliency
#         self.mode = mode
#         self.image_size = 224
#         self.class_num = 200
#
#         x = glob('data/CUB_200_2011/images/*/*.jpg')
#         import re
#         regex = re.compile(r'images/\d\d\d.')
#         y = [int(regex.search(xx).group()[7:-1]) for xx in x]
#         y = np.array(y).astype(int) - 1
#
#         cut = int(len(y) * 0.6)
#
#         combined = list(zip(x, y))
#         random.seed(327)
#         random.shuffle(combined)
#         x, y = zip(*combined)  # shuffle order
#
#         if mode == 'train' or mode == 'control' or mode == 'gt':
#             x, y = x[:cut], y[:cut]
#
#         elif mode == 'val':
#             x, y = x[cut:], y[cut:]
#
#         else:  # test, control
#             pass
#
#         self.x = x
#         self.y = y
#
#         if saliency:
#             if not mode == 'control' and not mode == 'gt':
#                 reuse = False
#                 if mode == 'val':
#                     reuse = True
#                 self.saliency_model = base_saliency_model(self.image_size, reuse=reuse)
#
#             if mode == 'gt':
#                 self.s = glob('data/CUB_200_2011/segmentations/*/*.png')
#
#
#         self.batch_size = batch_size
#         self.data_count = len(x)
#         self.num_batch = int(self.data_count / self.batch_size)
#         self.pointer = 0
#
#
#     # fixme:
#     def next_batch(self):
#         self.pointer = (self.pointer + 1) % self.num_batch
#
#         start_pos = self.pointer * self.batch_size
#
#         batch_images_dir = self.x[start_pos:start_pos + self.batch_size]
#
#         def to_rgb2(im):
#             w, h = im.shape
#             ret = np.empty((w, h, 3), dtype=np.uint8)
#             ret[:, :, :] = im[:, :, np.newaxis]
#             return ret
#
#         temp_bi = [np.array(Image.open(_d).resize([self.image_size, self.image_size]))
#                    if len(np.array(Image.open(_d)).shape) is 3
#                    else to_rgb2(np.array(Image.open(_d).resize([self.image_size, self.image_size])))
#                    for _d in batch_images_dir]
#         batch_images = np.array(temp_bi)
#
#         batch_labels = self.y[start_pos:start_pos + self.batch_size]
#
#         if self.saliency:
#             if self.mode == 'control':
#                 _batch_saliencies = [(np.random.randn(self.image_size, self.image_size)*255).astype('uint8')
#                                     for i in range(self.batch_size)]
#                 batch_saliencies = [np.reshape(b, [self.image_size, self.image_size, 1])
#                                     for b in _batch_saliencies]
#             elif self.mode == 'gt':
#
#                 batch_saliencies_dir = self.s[start_pos:start_pos + self.batch_size]
#                 temp_bs = [np.array(Image.open(_d).resize([self.image_size, self.image_size]))
#                            # if len(np.array(Image.open(_d)).shape) is 3
#                            # else to_rgb2(np.array(Image.open(_d).resize([self.image_size, self.image_size])))
#                            for _d in batch_saliencies_dir]
#                 _temp_bs = [np.reshape(b, [self.image_size, self.image_size, 1])
#                                     for b in temp_bs]
#                 batch_saliencies = np.array(_temp_bs)
#
#             else:
#                 batch_saliencies = self.saliency_model.get_saliency(batch_images)
#
#             batch_images = np.concatenate([batch_images, batch_saliencies], axis=3)
#
#         return batch_images, batch_labels

