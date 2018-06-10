"""
ISSUES:


"""


import tensorflow as tf
import tensorlayer as tl
slim = tf.contrib.slim
import numpy as np

# Overall
class SRAM(object):
    def __init__(self, config, inputs):

        self.config = config

        self.image_size = inputs.image_size
        self.image_shape = [self.image_size, self.image_size]
        self.class_num = inputs.class_num

        self.saliency = config.saliency
        if self.saliency:  # rgbs or rgb
            self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 4], name='x')
        else:
            self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3], name='x')

        self.y = tf.placeholder(tf.int32, shape=[None], name='y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self.glimpse_num = config.glimpse_num
        self.glimpse_size = config.glimpse_size
        self.hidden = config.hidden

        self.model_name = "SRAM-{}.model"

        self.build_model()
        self.build_loss_and_optimizer()
        self.merge_summary()


    def build_model(self):
        print('\nBuilding Model...')

        ## Encoder

        with tf.variable_scope("sram"):
            # Network Parts
            context_net = ContextNetwork(self.hidden)
            # fixme: self.image_size or self.glimpse_size ???
            st_net = MySpatialTransformerNetwork(self.hidden, out_size=[self.glimpse_size, self.glimpse_size],
                                                 glimpse_ratio=1.0)

            glimpse_net = GlimpseNetwork(self.glimpse_size, self.hidden)
            core_rnn = CoreRnn(self.hidden, st_net, glimpse_net)

            # initial context
            context = context_net(self.x, self.is_training, reuse=False)  # lstm_state_tuple: (c,h)
            glimpse, theta = st_net(self.x, context.h, self.is_training, reuse=False)

            _glimpse_patch = tf.nn.sigmoid(glimpse)  # sigmoid? tf.image.convert_image_dtype expect data range of [0,1)
            glimpse_patch = tf.image.convert_image_dtype(_glimpse_patch[:, :, :, :3], dtype=tf.float32)  #

            self.glimpse_patches = []  # glimpse 총 숫자는 지정한 glimpse_num + 1(처음 glimpse)
            self.glimpse_patches.append(glimpse_patch)  # initial glimpse

            if self.saliency:  # rgbs or rgb
                _saliency_glimpse_patch = tf.reshape(_glimpse_patch[:, :, :, -1], [-1, self.image_size, self.image_size, 1])
                saliency_glimpse_patch = tf.image.convert_image_dtype(_saliency_glimpse_patch, dtype=tf.float32)  #
                self.saliency_glimpse_patches = []
                self.saliency_glimpse_patches.append(saliency_glimpse_patch)

            glimpse_vec = glimpse_net(glimpse, theta, self.is_training, reuse=False)  # initial glimpse


            # core rnn
            lstm_1 = None
            lstm_2 = context

            for i in range(self.glimpse_num):
                print('Glimpse ', i)
                if i == 0:
                    is_first = True
                    reuse = False
                else:
                    is_first = False
                    reuse = True

                lstm_1, lstm_2, glimpse_vec, glimpse = core_rnn(
                    self.x, glimpse_vec, lstm_1, lstm_2, is_first, self.is_training, reuse)

                _glimpse_patch = tf.nn.sigmoid(glimpse)
                glimpse_patch = tf.image.convert_image_dtype(_glimpse_patch[:, :, :, :3], dtype=tf.float32)
                self.glimpse_patches.append(glimpse_patch)
                if self.saliency:
                    _saliency_glimpse_patch = tf.reshape(_glimpse_patch[:, :, :, -1], [-1, self.image_size, self.image_size, 1])
                    saliency_glimpse_patch = tf.image.convert_image_dtype(_saliency_glimpse_patch, dtype=tf.float32)

                    self.saliency_glimpse_patches.append(saliency_glimpse_patch)


            # try standardization before decoding
            self.z = lstm_1.outputs
            # mu, std = tf.nn.moments(self.z, axes=0)
            # self.z = (self.z - mu) / std

            classification_net = ClassificationNetwork(self.hidden, self.class_num)

            self.logits = classification_net(self.z, self.is_training, reuse=False)



    def build_loss_and_optimizer(self):

        # Decoder Loss
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.cross_entropy_loss = tf.reduce_mean(self.cross_entropy)
        self.accuracy = tf.metrics.accuracy(labels=self.y,
                                            predictions=tf.argmax(self.logits, axis=1))[1]
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.config.beta1)

        # self.train_op = self.optimizer.minimize(self.cross_entropy_loss)

        # fixme: grad is None?
        sram_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sram')
        gvs = self.optimizer.compute_gradients(self.cross_entropy_loss, var_list=sram_vars)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs)

    def merge_summary(self):

        # Summary
        self.image_sum = tf.summary.image("image", self.x[:, :, :, :3], max_outputs=4)

        if self.saliency:
            _x_saliency = self.x[:, :, :, -1]
            self.x_saliency = tf.reshape(_x_saliency, [-1, self.image_size, self.image_size, 1])
            self.saliency_sum = tf.summary.image("saliency", self.x_saliency, max_outputs=4)

        glimpses_concat = tf.concat(self.glimpse_patches, axis=0)
        self.glimpses_sum = tf.summary.image("glimpses", glimpses_concat, max_outputs=4)
        if self.saliency:
            saliency_glimpses_concat = tf.concat(self.saliency_glimpse_patches, axis=0)
            self.saliency_glimpses_sum = tf.summary.image("saliency_glimpses", saliency_glimpses_concat, max_outputs=4)

        self.cross_entropy_loss_sum = tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
        self.loss_sum = tf.summary.scalar("tot_loss", self.cross_entropy_loss)

        self.accuracy_sum = tf.summary.scalar("accuracy", self.accuracy)

        self.summary_merge = tf.summary.merge([self.image_sum, self.glimpses_sum, 
                                               self.cross_entropy_loss_sum, self.loss_sum, self.accuracy_sum]
                                              )
        if self.saliency:
            self.summary_merge = tf.summary.merge([self.image_sum, self.saliency_sum, 
                                                   self.glimpses_sum, self.saliency_glimpses_sum,
                                                   self.cross_entropy_loss_sum, self.loss_sum, self.accuracy_sum]
                                                  )


class SmallCnn(object):
    def __init__(self, hidden, n_filters, part):
        self.n_filters = n_filters
        self.part = part
        self.hidden = hidden

    def __call__(self, net, is_training, reuse, use_dilation=False):
        with tf.variable_scope("smallcnn_{}".format(self.part), reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001)):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.elu, updates_collections=None, is_training=is_training):

                    for i in range(len(self.n_filters)):
                        net = slim.conv2d(net, self.n_filters[i], 3, stride=1, padding='SAME',
                                          scope='{}_conv1_{}'.format(self.part, i))
                        net = slim.batch_norm(net, scope='{}_bn1_{}'.format(self.part, i))
                        net = slim.conv2d(net, self.n_filters[i], 3, stride=2, padding='SAME',
                                          scope='{}_conv2_{}'.format(self.part, i))
                        net = slim.batch_norm(net, scope='{}_bn2_{}'.format(self.part, i))

                    net = slim.flatten(net, scope='{}_flatten'.format(self.part))
                    net = slim.fully_connected(net, self.hidden, scope='{}_fc_{}'.format(self.part, i+1))
                    net = slim.batch_norm(net, scope='{}_bn_{}'.format(self.part, i+1))

        return net


### Encoder
class ContextNetwork(object):
    def __init__(self, hidden, n_filters=[16, 32, 64]):
        self.hidden = hidden
        self.n_filters = n_filters

    def __call__(self, x, is_training, reuse):
        print('Context Network')

        with tf.variable_scope("context", reuse=reuse):

            image_size = x.get_shape().as_list()[1]
            low_image = tf.image.resize_images(x, size=[int(image_size//4), int(image_size//4)])
            img_small_cnn = SmallCnn(self.hidden, n_filters=self.n_filters, part='image')

            net = img_small_cnn(low_image, is_training=is_training, reuse=reuse, use_dilation=False)

            with slim.arg_scope([slim.fully_connected], activation_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001)):

                    context_c = slim.fully_connected(net, self.hidden, scope='c_fc')
                    context_h = slim.fully_connected(net, self.hidden, scope='h_fc')
                    context = tf.contrib.rnn.LSTMStateTuple(context_c, context_h)

        return context

# define my own spatial transformer network based on SpatialTransformer2dAffineLayer
from transformer import transformer

# fixme: glimpse_size?
class MySpatialTransformerNetwork(object):
    """The :class:`SpatialTransformer2dAffineLayer` class is a
    `Spatial Transformer Layer <https://arxiv.org/abs/1506.02025>`_ for
    `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`_.

    Parameters
    -----------
    layer : a layer class with 4-D Tensor of shape [batch, height, width, channels]
    theta_layer : a layer class for the localisation network.
        In this layer, we will use a :class:`DenseLayer` to make the theta size to [batch, 6], value range to [0, 1] (via tanh).
    out_size : tuple of two ints.
        The size of the output of the network (height, width), the feature maps will be resized by this.

    References
    -----------
    - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`_
    - `TensorFlow/Models <https://github.com/tensorflow/models/tree/master/transformer>`_
    """

    def __init__(self, hidden, n_filters=[16, 32, 64], out_size=[40, 40], glimpse_ratio=1.0):
        # Layer.__init__(self, name='spatial_trans_2d_affine')
        self.hidden = hidden
        self.out_size = out_size
        self.glimpse_ratio = glimpse_ratio
        self.n_filters = n_filters

    def __call__(self, images, rnn2_outputs, is_training, reuse):
        print('Spatial Transformer Network')
        self.images = images

        with tf.variable_scope('spatial_trans_2d_affine', reuse=reuse) as vs:
            ## 1. make the localisation network to [batch, 6] via Flatten and Dense.
            # if self.theta_layer.outputs.get_shape().ndims > 2:
            #     self.theta_layer.outputs = flatten_reshape(self.theta_layer.outputs, 'flatten')

            _image = self.images
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001)):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.elu, updates_collections=None, is_training=is_training):

                    for i in range(len(self.n_filters)):
                        _image = slim.conv2d(_image, self.n_filters[i], 3, stride=1, padding='SAME',
                                          scope='iloc_conv1_' + str(i))
                        _image = slim.batch_norm(_image, scope='iloc_bn1' + str(i))
                        _image = slim.conv2d(_image, self.n_filters[i], 3, stride=2, padding='SAME',
                                          scope='iloc_conv2_' + str(i))

                    _image = slim.flatten(_image, scope='iloc_flatten')
                    _image = slim.fully_connected(_image, int(self.hidden/2), scope='iloc_fc_{}'.format(i+1))
                    _image = slim.batch_norm(_image, scope='iloc_bn_{}'.format(i+1))
                    _image = slim.fully_connected(_image, int(self.hidden/2), scope='iloc_fc_{}'.format(i+2))

            _rnn2 = rnn2_outputs
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training}
                                ):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.elu, updates_collections=None, is_training=is_training):
                    _rnn2 = slim.fully_connected(_rnn2, int(self.hidden/2), scope='rloc_fc1')
                    _rnn2 = slim.batch_norm(_rnn2, scope='rloc_bn1')
                    _rnn2 = slim.fully_connected(_rnn2, int(self.hidden/2), scope='rloc_fc2')

            self._theta = tf.concat([_image, _rnn2], axis=1) # concat above

            #self.theta_layer.outputs

            ## 2. To initialize the network to the identity transform init.
            # 2.1 W
            n_in = int(self._theta.get_shape()[-1])
            shape = (n_in, 6)
            W = tf.get_variable(name='W', initializer=tf.zeros(shape)) #, dtype=D_TYPE)
            # 2.2 b
            # fixme:
            identity = tf.constant(np.array([[self.glimpse_ratio, 0, 0], [0, self.glimpse_ratio, 0]]).astype('float32').flatten())
            b = tf.get_variable(name='b', initializer=identity) #, dtype=D_TYPE)
            # 2.3 transformation matrix
            self.theta = tf.nn.tanh(tf.matmul(self._theta, W) + 1) / 2 + b
            theta_constraint = tf.constant([1.0, 0, 1.0, 0, 1.0, 1.0])
            self.theta = tf.multiply(self.theta, theta_constraint)


            ## 3. Spatial Transformer Sampling
            # 3.1 transformation
            self.transformer = transformer()
            self.outputs = self.transformer(self.images, self.theta, out_size=self.out_size, reuse=reuse)
            # 3.2 automatically set batch_size and channels
            # e.g. [?, 40, 40, ?] --> [64, 40, 40, 1] or [64, 20, 20, 4]/ Hao Dong
            #
            fixed_batch_size = self.images.get_shape().with_rank_at_least(1)[0]
            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                from tensorflow.python.ops import array_ops
                batch_size = array_ops.shape(self.images)[0]
            size = self.images.get_shape().as_list()
            n_channels = self.images.get_shape().as_list()[-1]
            self.outputs = tf.reshape(self.outputs, shape=[batch_size, self.out_size[0], self.out_size[1], n_channels])


        return self.outputs, self.theta




class GlimpseNetwork(object):
    def __init__(self, glimpse_size, hidden, n_filters=[16, 32, 64]):
        self.n_filters = n_filters
        self.glimpse_size = glimpse_size
        self.hidden = hidden

    def __call__(self, glimpse, theta, is_training, reuse):

        print('Glimpse Network')
        with tf.variable_scope("glimpse", reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001)):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.elu, updates_collections=None, is_training=is_training):

                    net = glimpse

                    for i in range(len(self.n_filters)):
                        net = slim.conv2d(net, self.n_filters[i], 3, stride=1, padding='SAME',
                                          scope='conv1_' + str(i))
                        net = slim.conv2d(net, self.n_filters[i], 3, stride=2, padding='SAME',
                                          scope='conv2_' + str(i))
                        net = slim.batch_norm(net, scope='conv_bn_' + str(i))

                    net = slim.flatten(net)

                    # Apply different mlp for image/location
                    network_img_part = slim.fully_connected(net, self.hidden, scope='img_fc1')
                    network_loc_part = slim.fully_connected(theta, self.hidden, scope='loc_fc1')

                    # Combine
                    glimpse_vec = tf.multiply(network_img_part, network_loc_part)

        return glimpse_vec


class CoreRnn(object):
    def __init__(self, hidden, st_net, glimpse_net):
        self.hidden = hidden
        self.st_net = st_net
        self.glimpse_net = glimpse_net

    def __call__(self, x, glimpse_vec, _lstm_1, _lstm_2, is_first, is_training, reuse):
        # first lstm_2 initial state is context

        print('Core RNN')
        with tf.variable_scope("corernn", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            g = tl.layers.InputLayer(glimpse_vec, name='input')
            g = tl.layers.ReshapeLayer(g, [-1, 1, self.hidden], name='input1_reshape')  # input should be [batch_size, n_steps, n_features]

            # fixme: tensorlayer bug - no RNNLayer initial_state parameter
            if is_first:
                lstm_1_init_state = None
            else:
                lstm_1_init_state = _lstm_1.final_state

            lstm_1 = tl.layers.DynamicRNNLayer(g, cell_fn=tf.nn.rnn_cell.LSTMCell, n_hidden=self.hidden,
                                               initial_state=lstm_1_init_state, return_seq_2d=True, name='lstm_1')

            # tl bug - return_seq_2d=False doesn't work
            lstm_1_forinput = tl.layers.ReshapeLayer(lstm_1, [-1, 1, self.hidden], name='lstm_1_reshape')

            if is_first:  # case _lstm_2 is context
                lstm_2_init_state = _lstm_2
            else:
                lstm_2_init_state = _lstm_2.final_state

            lstm_2 = tl.layers.DynamicRNNLayer(lstm_1_forinput, cell_fn=tf.nn.rnn_cell.LSTMCell, n_hidden=self.hidden,
                                               initial_state=lstm_2_init_state, return_seq_2d=True, name='lstm_2')

        # CHECK - be careful with indent
        glimpse, theta = self.st_net(x, lstm_2.outputs, is_training, reuse=True)
        # fixme:
        # saliency_glimpse_patch = self.st_net.transformer(U=saliency, theta=theta,
        #                                              out_size=self.st_net.out_size, reuse=True)

        glimpse_vec = self.glimpse_net(glimpse, theta, is_training, reuse=True)

        return lstm_1, lstm_2, glimpse_vec, glimpse


#### Decoder


#### Classification
class ClassificationNetwork(object):
    def __init__(self, hidden, class_num):
        self.hidden = hidden
        self.class_num = class_num

    def __call__(self, z, is_training, reuse):
        print('Classification Network')
        with tf.variable_scope("classification", reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001)):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.elu, updates_collections=None, is_training=is_training):


                    net = z
                    net = slim.flatten(net, scope='flatten')
                    for i in range(3):
                        net = slim.fully_connected(net, self.hidden, scope='fc_{}'.format(i))
                        net = slim.batch_norm(net, scope='fc_bn_{}'.format(i))

                    net = slim.fully_connected(net, self.class_num, scope='logits')
                    logits = net

                    return logits





