import tensorflow as tf
import os, sys
import numpy as np

slim = tf.contrib.slim

class VGG():

    def __init__(self, config, inputs):

        self.config = config

        self.image_size = inputs.image_size
        self.image_shape = [self.image_size, self.image_size]
        self.class_num = inputs.class_num


        self.model_name = "VGG16.model"

        if config.saliency:  # rgbs or rgb
            self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 4], name='x')
        else:
            self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3], name='x')
        # self.y = tf.placeholder(tf.float32, shape=[None, self.class_num], name='y')
        self.y = tf.placeholder(tf.int32, shape=[None], name='y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')


        self.build_model()
        self.build_loss_and_optimizer()
        self.merge_summary()



    def build_model(self):

        # input
        net = self.x
        print(net.shape)

        if self.image_size == 224:  # vgg16
            filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            strides = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
            gp = 7
            hiddens = [4096, 4096]

        if self.image_size == 32:  # vgg16
            filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            strides = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
            gp = 1
            hiddens = [4096, 4096]

        # elif self.image_size == 32:
        #     filters = [32, 64, 64, 128, 128] #, 128, 128] #256, 256, 256, 256]
        #     strides = [2, 1, 2, 1, 2] #, 1, 2] #, 1, 2, 1, 2]
        #     gp = 4
        #     hiddens = [200, 200] #, 100]

        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.001)):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.elu, updates_collections=None, is_training=self.is_training):

                # vgg - conv
                for i in range(len(strides)):
                    net = slim.conv2d(net, filters[i], 3, stride=strides[i], padding='SAME', scope='conv{}'.format(i))
                    net = slim.batch_norm(net, scope='bn{}'.format(i))
                    print(net.shape)

                # vgg - fc, (=1x1 conv)

                net = slim.conv2d(net, hiddens[0], gp, stride=1, padding='VALID', scope='fc{}'.format(i+1))
                net = slim.batch_norm(net, scope='bn{}'.format(i + 1))
                net = slim.dropout(net, keep_prob=0.25, is_training=self.is_training, scope='dropout{}'.format(i + 1))
                print(net.shape)
                net = slim.conv2d(net, hiddens[1], 1, stride=1, padding='SAME', scope='fc{}'.format(i+2))
                net = slim.batch_norm(net, scope='bn{}'.format(i + 2))
                net = slim.dropout(net, keep_prob=0.25, is_training=self.is_training, scope='dropout{}'.format(i + 2))
                print(net.shape)

                net = slim.conv2d(net, self.class_num, 1, stride=1, padding='SAME', scope='fc{}'.format(i + 3))
                print(net.shape)

                self.logits = slim.flatten(net)
                print(self.logits.shape)

                # vgg - fc
                # net = slim.flatten(net)
                # for ii in range(len(hiddens)):
                #     net = slim.fully_connected(net, hiddens[ii], scope='fc{}'.format(i+1+ii))
                #     print(net.shape)

                # logits
                # self.logits = slim.fully_connected(net, self.class_num, scope='logits')
                # print(self.logits.shape)[

    def build_loss_and_optimizer(self):
        # self.cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits)
        # self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.y, axis=1),
        #                                     predictions=tf.argmax(self.logits, axis=1))[1]

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.cross_entropy_loss = tf.reduce_mean(self.cross_entropy)
        self.accuracy = tf.metrics.accuracy(labels=self.y,
                                            predictions=tf.argmax(self.logits, axis=1))[1]

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.config.beta1)
        #fixme:
        # gvs = self.optimizer.compute_gradients(self.cross_entropy_loss)
        # capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        # self.train_op = self.optimizer.apply_gradients(capped_gvs)

        self.train_op = self.optimizer.minimize(self.cross_entropy_loss)

    def merge_summary(self):

        summaries = []
        summaries += [tf.summary.image("image", self.x[:, :, :, :3], max_outputs=4)]
        if self.config.saliency:
            self.s = tf.reshape(self.x[:, :, :, -1], shape=[-1, self.image_size, self.image_size, 1])
            summaries += [tf.summary.image("saliency", self.s, max_outputs=4)]
        summaries += [tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)]
        summaries += [tf.summary.scalar("accuracy", self.accuracy)]

        self.summary_merge = tf.summary.merge(summaries)