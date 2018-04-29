import tensorflow as tf
import tensorlayer as tl
slim = tf.contrib.slim
import numpy as np

# Overall
class CRAM(object):
    def __init__(self, config, inputs):

        self.config = config
        self.is_training = config.is_training

        # Input Data
        self.orig_image, self.part_image, self.mask, self.mask_coord = \
            inputs['orig_image'], inputs['part_image'], inputs['mask'], inputs['mask_coord']
        self.masked_image = tf.multiply(self.orig_image, tf.ones_like(self.orig_image) - self.mask)
        # todo:

        self.glimpse_num = config.glimpse_num
        self.glimpse_size = config.glimpse_size
        self.hidden = config.hidden

        self.image_size = config.image_size
        self.image_shape = [self.image_size, self.image_size, 3]
        self.mask_ratio = config.mask_ratio

        self.model_name = "CRAM.model"

        task = config.task
        self.build_model(task)
        self.build_loss_and_optimizer(task)
        self.merge_summary(task)


    def build_model(self, task):
        print('Building Model...')

        ## Encoder

        # Network Parts
        context_net = ContextNetwork(self.hidden)
        st_net = MySpatialTransformerNetwork(self.hidden, out_size=[self.image_size, self.image_size], glimpse_ratio=1.0)

        glimpse_net = GlimpseNetwork(self.glimpse_size, self.hidden)
        core_rnn = CoreRnn(self.hidden, st_net, glimpse_net)

        # initial context
        context = context_net(self.masked_image, self.mask, self.is_training, reuse=False)  # lstm_state_tuple: (c,h)
        glimpse, theta, mytransformer = st_net(self.masked_image, self.mask, context.h, self.is_training, reuse=False)

        glimpse_patch = tf.nn.sigmoid(glimpse)
        glimpse_patch = tf.image.convert_image_dtype(glimpse_patch, dtype=tf.float32)  #

        self.glimpses = []  # glimpse 총 숫자는 지정한 glimpse_num + 1(처음 glimpse)
        self.glimpses.append(glimpse_patch)  # initial glimpse

        mask_glimpse = mytransformer(U=self.mask, theta=theta, out_size=[self.image_size, self.image_size], reuse=True)
        mask_glimpse_patch = tf.nn.sigmoid(mask_glimpse)
        mask_glimpse_patch = tf.image.convert_image_dtype(mask_glimpse_patch, dtype=tf.float32)  #

        self.glimpse_condition_correlation = []
        self.glimpse_condition_correlation.append(mask_glimpse_patch)

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

            lstm_1, lstm_2, glimpse_vec, glimpse_patch, mask_glimpse_patch = core_rnn(
                self.masked_image, self.mask, glimpse_vec, lstm_1, lstm_2, is_first, self.is_training, reuse)
            self.glimpses.append(glimpse_patch)

            self.glimpse_condition_correlation.append(mask_glimpse_patch)


        # try standardization before decoding
        self.z = lstm_1.outputs
        # mu, std = tf.nn.moments(self.z, axes=0)
        # self.z = (self.z - mu) / std

        completion_net = CompletionNetwork(self.hidden, self.image_size)
        self.completed_image, self.completed, self.generated = completion_net(self.z, self.masked_image, self.mask,
                                                                              self.mask_coord, self.mask_ratio,
                                                                              self.is_training, reuse=False)



        # Summary
        self.orig_image_sum = tf.summary.image("original_image", self.orig_image, max_outputs=4)
        self.masked_image_sum = tf.summary.image("masked_image", self.masked_image, max_outputs=4)
        self.recon_image_sum = tf.summary.image("reconstructed_image", self.generated, max_outputs=4)
        self.completed_part_image_sum = tf.summary.image("reconstructed_part", self.completed, max_outputs=4)
        self.completed_image_sum = tf.summary.image("completed_image", self.completed_image, max_outputs=4)

        glimpses_concat = tf.concat(self.glimpses, axis=1)
        self.glimpses_sum = tf.summary.image("glimpses", glimpses_concat, max_outputs=4)


    def build_loss_and_optimizer(self, task):
        ## Loss


        # Encoder Loss
        """
        glimpse should be
        1. diversity
        2. near binary condition
        """

        # Constraints to make glimpse different from each other
        diverse_glimpse_loss = None

        # loss near condition
        conditional_glimpse_losses = []
        for m_g in self.glimpse_condition_correlation:
            conditional_glimpse_losses.append(1 - tf.reduce_mean(m_g))
        self.conditional_glimpse_loss = tf.reduce_mean(conditional_glimpse_losses)


        # Decoder Loss

        # Reconstruction Loss - l1
        recon_difference = tf.subtract(self.orig_image, self.completed_image)
        self.recon_loss = tf.reduce_sum(tf.abs(recon_difference))

        self.gradients = tf.gradients(self.recon_loss, tf.trainable_variables())

        # self.gradients

        ## Critic
        local_critic = LocalCritic()
        global_critic = GlobalCritic()

        local_D_real, local_D_real_logits = local_critic(self.part_image, self.is_training, reuse=False)
        local_D_fake, local_D_fake_logits = local_critic(self.completed, self.is_training, reuse=True)
        global_D_real, global_D_real_logits = global_critic(self.orig_image, self.is_training, reuse=False)
        global_D_fake, global_D_fake_logits = global_critic(self.generated, self.is_training, reuse=True)

        local_D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=local_D_real_logits, labels=tf.ones_like(local_D_real)))
        local_D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=local_D_fake_logits, labels=tf.zeros_like(local_D_fake)))
        self.local_D_loss = local_D_loss_real + local_D_loss_fake

        global_D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=global_D_real_logits, labels=tf.ones_like(global_D_real)))
        global_D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=global_D_fake_logits, labels=tf.zeros_like(global_D_fake)))
        self.global_D_loss = global_D_loss_real + global_D_loss_fake


        self.local_G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=local_D_fake_logits, labels=tf.ones_like(local_D_fake)))
        self.global_G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=global_D_fake_logits, labels=tf.ones_like(global_D_fake)))


        self.D_loss = self.local_D_loss + self.global_D_loss

        alpha = 0.01
        beta = 0.1
        self.G_loss = (self.local_G_loss + self.global_G_loss)
        self.G_loss_tot = alpha * self.recon_loss + beta * self.conditional_glimpse_loss + self.G_loss


        ## Optimizer
        self.recon_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
            .minimize(self.recon_loss)

        self.d_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
            .minimize(self.D_loss)
        self.g_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
            .minimize(self.G_loss_tot)


        # Summary
        self.recon_loss_sum = tf.summary.scalar("recon_loss", self.recon_loss)
        self.conditional_glimpse_loss_sum = tf.summary.scalar("conditional_glimpse_loss", self.conditional_glimpse_loss)

        self.d_loss_sum = tf.summary.scalar("d_loss", self.D_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)

        self.g_loss_tot_sum = tf.summary.scalar("g_loss_including_recon", self.G_loss_tot)

        self.recon_grad_sum = [tf.summary.histogram('recon_grad'+str(i), tf.norm(self.gradients[i]))
                               for i in range(len(self.gradients))]


    def merge_summary(self):
        self.temp_summary = tf.summary.merge([self.masked_image_sum, self.glimpses_sum])

        self.all_summary = tf.summary.merge([self.orig_image_sum,
                                             self.masked_image_sum,
                                             self.recon_image_sum,
                                             self.completed_part_image_sum,
                                             self.completed_image_sum,
                                             self.glimpses_sum,

                                             self.recon_loss_sum,
                                             self.conditional_glimpse_loss_sum,
                                             self.d_loss_sum,
                                             self.g_loss_sum,
                                             self.g_loss_tot_sum
                                             ] + self.recon_grad_sum)






#
# Networks
#


class SmallCnn(object):
    def __init__(self, hidden, n_filters, part):
        self.n_filters = n_filters
        self.part = part
        self.hidden = hidden

    def __call__(self, net, is_training, reuse, use_dilation=False):
        with tf.variable_scope("smallcnn_{}".format(self.part), reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            for i in range(len(self.n_filters)):
                if use_dilation:
                    net = tl.layers.AtrousConv2dLayer(net, n_filter=self.n_filters[i], filter_size=(3, 3), rate=2,
                                                      padding='SAME', name=('{}_astconv' + str(i)).format(self.part))
                else:
                    net = tl.layers.Conv2d(net, n_filter=self.n_filters[i], filter_size=(3, 3), strides=(1, 1),
                                           padding='SAME', name=('{}_conv' + str(i)).format(self.part))
                net = tl.layers.BatchNormLayer(net, act=tf.nn.elu, is_train=is_training,
                                               name=('{}_conv_bn' + str(i)).format(self.part))
                net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME',
                                          name=('{}_maxpool' + str(i)).format(self.part))

            net = tl.layers.FlattenLayer(net, name='{}_flatten'.format(self.part))
            net = tl.layers.DenseLayer(net, n_units=self.hidden, name='{}_fc1'.format(self.part))
            net = tl.layers.BatchNormLayer(net, act=tf.nn.elu, name='{}_fc_bn1'.format(self.part))

        return net


### Encoder
class ContextNetwork(object):
    def __init__(self, hidden, n_filters=[16, 32, 64]):
        self.hidden = hidden
        self.n_filters = n_filters

    def __call__(self, masked_image, mask, is_training, reuse):
        """

        :param image:
        :param mask:
        :param is_training:
        :return: layer
        """
        with tf.variable_scope("context", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            print('Context Network')
            image_size = masked_image.get_shape().as_list()[1]
            # context for image
            # Downsample the image
            # fixme: downsample rate, n_filters
            low_image = tf.image.resize_images(masked_image, size=[int(image_size//4), int(image_size//4)])
            context_image_part = tl.layers.InputLayer(low_image, name='context_image_input')
            img_small_cnn = SmallCnn(self.hidden, n_filters=self.n_filters, part='context_image')
            context_image_part = img_small_cnn(context_image_part, is_training=is_training, reuse=reuse, use_dilation=True)

            context_mask_part = tl.layers.InputLayer(mask, name='context_mask_input')
            # check - mask part is using different cnn(feature extractor) with image part
            mask_small_cnn = SmallCnn(self.hidden, n_filters=self.n_filters, part='context_mask')
            context_mask_part = mask_small_cnn(context_mask_part, is_training=is_training, reuse=reuse, use_dilation=True)

            net = tl.layers.ConcatLayer([context_image_part, context_mask_part], 1, name='context_concat')

            context_c = tl.layers.DenseLayer(net, n_units=self.hidden, name='context_c_fc')
            context_h = tl.layers.DenseLayer(net, n_units=self.hidden, name='context_h_fc')

            context = tf.contrib.rnn.LSTMStateTuple(context_c.outputs, context_h.outputs)

        return context


# define my own spatial transformer network based on SpatialTransformer2dAffineLayer
# from tensorlayer import Layer
from transformer import transformer

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

    def __init__(self, hidden, out_size=[40, 40], glimpse_ratio=1.0):
        # Layer.__init__(self, name='spatial_trans_2d_affine')
        self.hidden = hidden
        self.out_size = out_size
        self.glimpse_ratio = glimpse_ratio

        print("  [TL] SpatialTransformer2dAffineLayer %s: in_size:%s out_size:%s" %
              ('spatial_trans_2d_affine', self.inputs.get_shape().as_list(), out_size))

    def __call__(self, images, clues, rnn2_outputs, is_training, reuse):
        print('Spatial Transformer Network')
        self.inputs = images
        self.clues = clues

        with tf.variable_scope('spatial_trans_2d_affine', reuse=reuse) as vs:
            ## 1. make the localisation network to [batch, 6] via Flatten and Dense.
            # if self.theta_layer.outputs.get_shape().ndims > 2:
            #     self.theta_layer.outputs = flatten_reshape(self.theta_layer.outputs, 'flatten')

            _image = self.inputs
            n_filters = [8, 16, 32]
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.elu,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training},
                                ):
                for i in range(len(n_filters)):
                    _image = slim.conv2d(_image, n_filters[i], 5, stride=2, padding='SAME',
                                      scope='iloc_conv' + str(i))

                _image = slim.fully_connected(_image, int(self.hidden/2), scope='iloc_fc1')
            _image = slim.fully_connected(_image, 10, scope='iloc_fc2')  # no activation


            _clue = self.clues
            n_filters = [8, 16, 32]
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.elu,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training},
                                ):
                for i in range(len(n_filters)):
                    _clue = slim.conv2d(_clue, n_filters[i], 5, stride=2, padding='SAME',
                                         scope='cloc_conv' + str(i))

                _clue = slim.fully_connected(_clue, int(self.hidden/2), scope='cloc_fc1')
            _clue = slim.fully_connected(_clue, 10, scope='cloc_fc2')  # no activation


            _rnn2 = rnn2_outputs
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training}
                                ):
                _rnn2 = slim.fully_connected(_rnn2, int(self.hidden/2), scope='rloc_fc1')
            _rnn2 = slim.fully_connected(_rnn2, 10, scope='rloc_fc2')

            self._theta = tf.concat([_image, _clue, _rnn2]) # concat above

            #self.theta_layer.outputs

            ## 2. To initialize the network to the identity transform init.
            # 2.1 W
            n_in = int(self._theta.get_shape()[-1])
            shape = (n_in, 6)
            W = tf.get_variable(name='W', initializer=tf.zeros(shape)) #, dtype=D_TYPE)
            # 2.2 b
            identity = tf.constant(np.array([[self.glimpse_ratio, 0, 0], [0, self.glimpse_ratio, 0]]).astype('float32').flatten())
            b = tf.get_variable(name='b', initializer=identity) #, dtype=D_TYPE)
            # 2.3 transformation matrix
            self.theta = tf.nn.tanh(tf.matmul(self._theta, W) + 1) / 2 + b
            theta_constraint = tf.constant([1.0, 0, 1.0, 0, 1.0, 1.0])
            self.theta = tf.multiply(self.theta, theta_constraint)


            ## 3. Spatial Transformer Sampling
            # 3.1 transformation
            self.transformer = transformer()
            self.outputs = self.transformer(U=self.inputs, theta=self.theta, out_size=self.out_size, reuse=reuse)
            # 3.2 automatically set batch_size and channels
            # e.g. [?, 40, 40, ?] --> [64, 40, 40, 1] or [64, 20, 20, 4]/ Hao Dong
            #
            fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                from tensorflow.python.ops import array_ops
                batch_size = array_ops.shape(self.inputs)[0]
            size = self.inputs.get_shape().as_list()
            n_channels = self.inputs.get_shape().as_list()[-1]
            self.outputs = tf.reshape(self.outputs, shape=[batch_size, self.out_size[0], self.out_size[1], n_channels])

        try:  # For TF12 and later
            TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
        except:  # For TF11 and before
            TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.VARIABLES

        variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        self.all_params = variables

        return self.outputs, self.theta, self.transformer

        #     ## 4. Get all parameters
        #    variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        #
        #
        # # fixme
        # if not reuse:
        #     ## fixed
        #     self.all_layers = list(layer.all_layers)
        #     self.all_params = list(layer.all_params)
        #     self.all_drop = dict(layer.all_drop)
        #
        #     ## theta_layer
        #     self.all_layers.extend(theta_layer.all_layers)
        #     self.all_params.extend(theta_layer.all_params)
        #     self.all_drop.update(theta_layer.all_drop)
        #
        #     ## this layer
        #     self.all_layers.extend([self.outputs])
        #     self.all_params.extend(variables)



class GlimpseNetwork(object):
    def __init__(self, glimpse_size, hidden, n_filters=[16, 32, 64]):
        self.n_filters = n_filters
        self.glimpse_size = glimpse_size
        self.hidden = hidden

    def __call__(self, glimpse_patch, theta, is_training, reuse):
        """

        :param theta_layer: transform matrix(layer)
        :return:
        """
        with tf.variable_scope("gl", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            print('Glimpse Network')
            # Image(conv) Result
            net = tl.layers.InputLayer(glimpse_patch, name='glimpse_input')

            for i in range(len(self.n_filters)):
                net = tl.layers.Conv2d(net, n_filter=self.n_filters[i], filter_size=(3, 3), strides=(1, 1),
                                       padding='SAME', name='glimpse_conv' + str(i))
                net = tl.layers.BatchNormLayer(net, act=tf.nn.elu, is_train=is_training,
                                               name='glimpse_conv_bn' + str(i))
                # if i+1 % 2 == 0:  # pooling after even number layer
                net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME',
                                          name='glimpse_maxpool' + str(i))

            # Apply different mlp for image/location
            network_img_part = tl.layers.FlattenLayer(net, name='glimpse_img_flatten')
            network_img_part = tl.layers.DenseLayer(network_img_part, n_units=self.hidden, name='glimpse_img_fc1')
            network_img_part = tl.layers.BatchNormLayer(network_img_part, act=tf.nn.elu, is_train=is_training,
                                                        name='glimpse_img_fc_bn1')

            # Location result
            theta_layer = tl.layers.InputLayer(theta, name='theta_input_layer')
            network_loc_part = tl.layers.DenseLayer(theta_layer, n_units=self.hidden, name='glimpse_loc_fc2')
            network_loc_part = tl.layers.BatchNormLayer(network_loc_part, act=tf.nn.elu, is_train=is_training,
                                                        name='glimpse_loc_fc_bn')
            # Combine
            glimpse_vec = tf.multiply(network_img_part.outputs, network_loc_part.outputs)

        return glimpse_vec


class CoreRnn(object):
    def __init__(self, hidden, st_net, glimpse_net):
        self.hidden = hidden
        self.st_net = st_net
        self.glimpse_net = glimpse_net

    def __call__(self, masked_image, mask, glimpse_vec, _lstm_1, _lstm_2, is_first, is_training, reuse):
        # first lstm_2 initial state is context
        """

        :param masked_image:
        :param mask:
        :param glimpse: vec - tensor
        :param _lstm_1: layer
        :param _lstm_2: layer
        :param emission_net:
        :param glimpse_net:
        :return: layer
        """

        self.image_size = masked_image.shape[1]

        print('Core RNN')
        with tf.variable_scope("corernn", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            x = tl.layers.InputLayer(glimpse_vec, name='core_input')
            x = tl.layers.DenseLayer(x, n_units=self.hidden, name='core_fc1')
            x = tl.layers.ReshapeLayer(x, [-1, 1, self.hidden], name='core_input1_reshape')  # input should be [batch_size, n_steps, n_features]

            # fixme: tensorlayer 버그 - RNNLayer initial_state parameter에 없다고 나옴
            if is_first:
                lstm_1_init_state = None
            else:
                lstm_1_init_state = _lstm_1.final_state
            lstm_1 = tl.layers.DynamicRNNLayer(x, cell_fn=tf.nn.rnn_cell.LSTMCell, n_hidden=self.hidden,
                                               initial_state=lstm_1_init_state, return_seq_2d=True, name='core_lstm_1')

            # return_seq_2d=False 하면 안 되는 버그 있어서 한 번 돌아감
            lstm_1_forinput = tl.layers.ReshapeLayer(lstm_1, [-1, 1, self.hidden], name='core_lstm_1_reshape')

            if is_first:  # case _lstm_2 is context
                lstm_2_init_state = _lstm_2
            else:
                lstm_2_init_state = _lstm_2.final_state
            lstm_2 = tl.layers.DynamicRNNLayer(lstm_1_forinput, cell_fn=tf.nn.rnn_cell.LSTMCell, n_hidden=self.hidden,
                                               initial_state=lstm_2_init_state, return_seq_2d=True, name='core_lstm_2')

        # CHECK - be careful with indent
        glimpse_patch, theta, mytransformer = self.st_net(masked_image, mask, lstm_2.outputs, is_training, reuse=True)
        mask_glimpse_patch = mytransformer(mask, theta, out_size=[self.image_size, self.image_size], reuse=True)

        glimpse_vec = self.glimpse_net(masked_image, mask, theta, is_training, reuse=True)

        return lstm_1, lstm_2, glimpse_vec, glimpse_patch, mask_glimpse_patch





















#### Decoder


class ClassificationNetwork(object):
    def __init__(self, hidden):
        self.hidden = hidden

    def __call__(self, z, is_training, reuse):
        print('Classification Network')
        with tf.variable_scope("classification", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net = tl.layers.InputLayer(z, name='cls_input')
            net = tl.layers.FlattenLayer(net, name='cls_flatten')
            net = tl.layers.DenseLayer(net, n_units=self.hidden, name='cls_fc1')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.elu, name='cls_fc_bn1')
            net = tl.layers.DenseLayer(net, n_units=5, name='cls_logits')

            logits = net.outputs

            return logits


class CompletionNetwork(object):
    def __init__(self, hidden, image_size):
        self.hidden = hidden
        self.image_size = image_size

    def __call__(self, z, masked_image, mask, mask_coord, mask_ratio, is_training, reuse):
        """

        :param z:
        :param masked_image:
        :param mask:
        :param mask_coord:
        :param mask_ratio:
        :param is_training:
        :param reuse:
        :return:
        """
        print('Completion Network')
        with tf.variable_scope("completion", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            completed_image_network = tl.layers.InputLayer(z, name='completion_input')  # dimension cram.hidden [100]
            completed_image_network = tl.layers.ReshapeLayer(completed_image_network, [-1, 1, 1, self.hidden],
                                                             name='completion_input_reshape')

            # deconvolution
            n_filters = 128
            idx = 1

            completed_image_size = completed_image_network.outputs.get_shape().as_list()[1]
            while completed_image_size < int(self.image_size//2):
                # 버그 - tl.layers.DeConv2d n_filters 아니고 n_out_channel
                completed_image_network = tl.layers.DeConv2d(
                    completed_image_network, n_out_channel=n_filters, filter_size=(3, 3),
                    out_size=(int(completed_image_size*2), int(completed_image_size*2)),
                    strides=(2, 2), padding='SAME', name='complete_transconv' + str(idx))
                completed_image_network = tl.layers.BatchNormLayer(
                    completed_image_network, act=tf.nn.elu, is_train=is_training,
                    name='complete_transconv_bn' + str(idx))
                idx += 1
                completed_image_size = completed_image_network.outputs.get_shape().as_list()[1]
                if n_filters > 4:
                    n_filters = int(n_filters // 2)
                # if n_filters > 32:
                #     n_filters = int(n_filters - 32)
                # else:


            completed_image_network = tl.layers.DeConv2d(
                completed_image_network, n_out_channel=3, filter_size=(3, 3),
                out_size=(int(completed_image_size * 2), int(completed_image_size * 2)),
                strides=(2, 2), padding='SAME', name='complete_transconv' + str(idx))
            completed_image_network = tl.layers.BatchNormLayer(
                completed_image_network, is_train=is_training, name='complete_transconv_bn' + str(idx))

            # todo
            # check
            generated = tf.nn.tanh(completed_image_network.outputs) + 1
            # generated = tf.nn.elu(completed_image_network.outputs)
            generated = tf.image.convert_image_dtype(generated, dtype=tf.float32)  #

            completed_image = tf.add(masked_image, tf.multiply(generated, mask))

            # fixme -- need to use random_crop
            mask_size = int(self.image_size * mask_ratio)
            mx = int((self.image_size - mask_size) / 2)
            completed = tf.image.crop_to_bounding_box(completed_image, mx, mx, mask_size, mask_size)
            completed = tf.image.resize_images(completed, [int(self.image_size // 4), int(self.image_size // 4)])
            completed = tf.image.convert_image_dtype(completed, dtype=tf.float32)


            # Random_crop
            #completed = Utils.crop_to_multiple_bounding_box(completed_image, mask_coord)


        return completed_image, completed, generated


# todo: WGAN
class LocalCritic(object):
    def __init__(self, n_filters=[16, 32, 64, 128]):
        self.n_filters = n_filters

    def __call__(self, completed, is_training, reuse):
        with tf.variable_scope("localcritic", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            print('Local Critic')
            net = tl.layers.InputLayer(completed, name='local_critic_input')
            for i in range(len(self.n_filters)):
                idx = i + 1
                net = tl.layers.Conv2d(net, n_filter=self.n_filters[i], filter_size=(5, 5), strides=(2, 2), padding='SAME',
                                       name='local_critic_conv' + str(idx))
                net = tl.layers.BatchNormLayer(net, act=tf.nn.elu, is_train=is_training,
                                               name='local_critic_conv_bn' + str(idx))
                # net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME',
                #                           name='local_critic_maxpool' + str(idx))

            net = tl.layers.FlattenLayer(net)
            net = tl.layers.DenseLayer(net, n_units=100)
            net = tl.layers.DenseLayer(net, n_units=1, name='local_critic_fc1')
            logits = net.outputs

        return tf.nn.sigmoid(logits), logits


class GlobalCritic(object):
    def __init__(self, n_filters=[16, 32, 64, 128, 256]):
        self.n_filters = n_filters

    def __call__(self, completed_image, is_training, reuse):
        with tf.variable_scope("globalcritic", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            print('Global Critic')
            net = tl.layers.InputLayer(completed_image, name='global_critic_input')
            for i in range(len(self.n_filters)):
                idx = i + 1
                net = tl.layers.Conv2d(net, n_filter=self.n_filters[i], filter_size=(5, 5), strides=(2, 2), padding='SAME',
                                       name='global_critic_conv' + str(idx))
                net = tl.layers.BatchNormLayer(net, act=tf.nn.elu, is_train=is_training,
                                               name='global_critic_conv_bn' + str(idx))
                # net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME',
                #                           name='global_critic_maxpool' + str(idx))

            net = tl.layers.FlattenLayer(net)
            net = tl.layers.DenseLayer(net, n_units=100)
            net = tl.layers.DenseLayer(net, n_units=1, name='global_critic_fc1')
            logits = net.outputs

        return tf.nn.sigmoid(logits), logits





