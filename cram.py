"""
ISSUES:

1. consider number of hidden based on computation power
2. match shape between context output and lstm_2 -- 100
3. xavier initialization


4. glimpse visualization

* using dilated convolution

dataset api
add task

"""


import tensorflow as tf
import tensorlayer as tl


# Overall
class CRAM(object):
    def __init__(self, config, inputs):

        self.config = config
        self.is_training = config.is_training

        # Input Data
        self.orig_image, self.part_image, self.mask, self.mask_coord = \
            inputs['orig_image'], inputs['part_image'], inputs['mask'], inputs['mask_coord']
        self.masked_image = tf.multiply(self.orig_image, tf.ones_like(self.orig_image) - self.mask)

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
        stgl_net = ST_GlimpseNetwork(self.glimpse_size, self.hidden)
        emission_net = EmissionNetwork(self.hidden)
        core_rnn = CoreRnn(self.hidden, emission_net, stgl_net)
        # solution_net = SolutionNetwork()

        # initial context
        context = context_net(self.masked_image, self.mask, self.is_training, reuse=False) # lstm_state_tuple: (c,h)
        theta = emission_net(context.h, self.mask, self.is_training, reuse=False)

        self.glimpses = []
        glimpse_vec, glimpse_patch = stgl_net(self.masked_image, theta, self.is_training, reuse=False)  # initial glimpse
        self.glimpses.append(glimpse_patch)  # initial glimpse

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
            lstm_1, lstm_2, glimpse_vec, glimpse_patch = core_rnn(self.masked_image, self.mask, glimpse_vec, lstm_1, lstm_2,
                                                                  is_first, self.is_training, reuse)
            self.glimpses.append(glimpse_patch)

        # solution = solution_net(lstm_1.outputs, self.is_training, reuse=False)
        # remove solution network


        # try standardization before decoding
        self.z = lstm_1.outputs
        # mu, std = tf.nn.moments(self.z, axes=0)
        # self.z = (self.z - mu) / std

        if task == 'inpainting':

            ## Decoder

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
        # Reconstruction Loss
        recon_difference = tf.subtract(self.orig_image, self.generated)
        self.recon_loss = tf.reduce_mean(tf.nn.l2_loss(tf.multiply(recon_difference, self.mask)))

        self.gradients = tf.gradients(self.recon_loss, tf.trainable_variables())
        # self.gradients_norm = tf.norm(self.gradients)

        # self.gradients

        ## Critic
        local_critic = LocalCritic()
        global_critic = GlobalCritic()

        if task == 'inpainting':
            use_wgan = self.config.use_wgan

        if not use_wgan:
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

            self.D_loss = self.local_D_loss + self.global_D_loss

            self.local_G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=local_D_fake_logits, labels=tf.ones_like(local_D_fake)))
            self.global_G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=global_D_fake_logits, labels=tf.ones_like(global_D_fake)))

            alpha = 0.01
            self.G_loss = (self.local_G_loss + self.global_G_loss)
            self.G_loss_with_recon = alpha*self.recon_loss + self.G_loss


            ## Optimizer
            self.recon_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                .minimize(self.recon_loss)

            self.d_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                .minimize(self.D_loss)
            self.g_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                .minimize(self.G_loss_with_recon)

        # todo:
        else:
            self.wgan_loss = None


        # Summary
        self.recon_loss_sum = tf.summary.scalar("recon_loss", self.recon_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.D_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)
        self.g_loss_with_recon_sum = tf.summary.scalar("g_loss_including_recon", self.G_loss_with_recon)

        self.recon_grad_sum = [tf.summary.histogram('recon_grad'+str(i), tf.norm(self.gradients[i]))
                               for i in range(len(self.gradients))]


    def merge_summary(self, task):
        if task == 'inpainting':
            self.all_summary = tf.summary.merge([self.orig_image_sum,
                                                 self.masked_image_sum,
                                                 self.recon_image_sum,
                                                 self.completed_part_image_sum,
                                                 self.completed_image_sum,
                                                 self.glimpses_sum,

                                                 self.recon_loss_sum,
                                                 self.d_loss_sum,
                                                 self.g_loss_sum,
                                                 self.g_loss_with_recon_sum
                                                 ] + self.recon_grad_sum)









class SmallCnn(object):
    def __init__(self, hidden, n_filters, part):
        self.n_filters = n_filters
        self.part = part
        self.hidden = hidden

    def __call__(self, net, is_training, reuse):
        with tf.variable_scope("smallcnn_{}".format(self.part), reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            for i in range(len(self.n_filters)):
                net = tl.layers.Conv2d(net, n_filter=self.n_filters[i], filter_size=(3, 3), strides=(1, 1), padding='SAME',
                                       name=('{}_conv' + str(i)).format(self.part))
                net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_training,
                                               name=('{}_conv_bn' + str(i)).format(self.part))
                net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME',
                                          name=('{}_maxpool' + str(i)).format(self.part))

            net = tl.layers.FlattenLayer(net, name='{}_flatten'.format(self.part))
            net = tl.layers.DenseLayer(net, n_units=self.hidden, name='{}_fc1'.format(self.part))
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, name='{}_fc_bn1'.format(self.part))

        return net


### Encoder
class ContextNetwork(object):
    def __init__(self, hidden, n_filters=[16, 32, 64]):
        self.hidden = hidden
        self.n_filters = n_filters

    def __call__(self, image, mask, is_training, reuse):
        """

        :param image:
        :param mask:
        :param is_training:
        :return: layer
        """
        with tf.variable_scope("context", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            print('Context Network')
            image_size = image.get_shape().as_list()[1]
            # context for image
            # Downsample the image
            # fixme: downsample rate, n_filters
            low_image = tf.image.resize_images(image, size=[int(image_size//4), int(image_size//4)])
            context_image_part = tl.layers.InputLayer(low_image, name='context_image_input')
            img_small_cnn = SmallCnn(self.hidden, n_filters=self.n_filters, part='context_image')
            context_image_part = img_small_cnn(context_image_part, is_training=is_training, reuse=reuse)

            context_mask_part = tl.layers.InputLayer(mask, name='context_mask_input')
            # check - mask part is using different cnn(feature extractor) with image part
            mask_small_cnn = SmallCnn(self.hidden, n_filters=self.n_filters, part='context_mask')
            context_mask_part = mask_small_cnn(context_mask_part, is_training=is_training, reuse=reuse)

            net = tl.layers.ConcatLayer([context_image_part, context_mask_part], 1, name='context_concat')

            context_c = tl.layers.DenseLayer(net, n_units=self.hidden, name='context_c_fc')
            context_h = tl.layers.DenseLayer(net, n_units=self.hidden, name='context_h_fc')

            context = tf.contrib.rnn.LSTMStateTuple(context_c.outputs, context_h.outputs)

        return context


class ST_GlimpseNetwork(object):
    def __init__(self, glimpse_size, hidden, n_filters=[16, 32, 64]):
        self.n_filters = n_filters
        self.glimpse_size = glimpse_size
        self.hidden = hidden

    def __call__(self, masked_image, theta_layer, is_training, reuse):
        """

        :param theta_layer: transform matrix(layer)
        :return:
        """
        with tf.variable_scope("stgl", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            print('STGL Network')
            # Image(conv) Result
            # Spatial transformation
            net = tl.layers.InputLayer(masked_image, name='glimpse_input')
            net = tl.layers.SpatialTransformer2dAffineLayer(net, theta_layer=theta_layer,
                                                            out_size=[self.glimpse_size, self.glimpse_size],
                                                            name='glimpse_st')

            glimpse_patch = tf.nn.sigmoid(net.outputs)
            glimpse_patch = tf.image.convert_image_dtype(glimpse_patch, dtype=tf.float32)  #


            for i in range(len(self.n_filters)):
                net = tl.layers.Conv2d(net, n_filter=self.n_filters[i], filter_size=(3, 3), strides=(1, 1),
                                       padding='SAME', name='glimpse_conv' + str(i))
                net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_training,
                                               name='glimpse_conv_bn' + str(i))
                # if i+1 % 2 == 0:  # pooling after even number layer
                net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME',
                                          name='glimpse_maxpool' + str(i))

            # Apply different mlp for image/location
            network_img_part = tl.layers.FlattenLayer(net, name='glimpse_img_flatten')
            network_img_part = tl.layers.DenseLayer(network_img_part, n_units=self.hidden, name='glimpse_img_fc1')
            network_img_part = tl.layers.BatchNormLayer(network_img_part, act=tf.nn.relu, is_train=is_training,
                                                        name='glimpse_img_fc_bn1')

            # Location result
            network_loc_part = tl.layers.DenseLayer(theta_layer, n_units=self.hidden, name='glimpse_loc_fc2')
            network_loc_part = tl.layers.BatchNormLayer(network_loc_part, act=tf.nn.relu, is_train=is_training,
                                                        name='glimpse_loc_fc_bn')
            # Combine
            glimpse_vec = tf.multiply(network_img_part.outputs, network_loc_part.outputs)

        return glimpse_vec, glimpse_patch


class EmissionNetwork(object):
    def __init__(self, hidden, n_filters=[16, 32, 64]):
        self.hidden = hidden
        self.n_filters = n_filters

    def __call__(self, lstm_result, mask, is_training, reuse):
        """

        :param lstm_result: tensor
        :param condition:
        :return: layer
        """
        with tf.variable_scope("emission", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            print('Emission Network')
            net = tl.layers.InputLayer(lstm_result, name='emission_input')

            # conv Condition -- Image_mask
            emssion_mask_small_cnn = SmallCnn(self.hidden, n_filters=self.n_filters, part='emission_mask')
            mask = tl.layers.InputLayer(mask, name='emission_mask_input')
            mask = emssion_mask_small_cnn(mask, is_training=is_training, reuse=reuse)

            # concat condition with network
            net = tl.layers.ConcatLayer([mask, net], 1, name='emission_concat')

            net = tl.layers.DenseLayer(net, n_units=self.hidden, name='emission_fc1')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_training, name='emission_fc_bn1')

            #check: theta should be activated by tf.nn.tanh
            net = tl.layers.DenseLayer(net, n_units=6, act=tf.nn.tanh, name='emission_fc2')

            # sx, sy, tx, ty - attention
            _theta = net.outputs
            theta_constraint = tf.constant([1, 0, 1, 0, 1, 1])
            theta = tf.multiply(_theta, theta_constraint)
            theta_layer = tl.layers.InputLayer(theta, name='theta_layer')  # output format should be layer

        return theta_layer


class CoreRnn(object):
    def __init__(self, hidden, emission_net, stgl_net):
        self.hidden = hidden
        self.emission_net = emission_net
        self.stgl_net = stgl_net

    def __call__(self, masked_image, mask, glimpse_vec, _lstm_1, _lstm_2, is_first, is_training, reuse):
        # first lstm_2 initial state is context
        """

        :param masked_image:
        :param mask:
        :param glimpse: vec - tensor
        :param _lstm_1: layer
        :param _lstm_2: layer
        :param emission_net:
        :param stgl_net:
        :return: layer
        """
        print('Core RNN')
        with tf.variable_scope("corernn", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            x = tl.layers.InputLayer(glimpse_vec, name='core_input')
            x = tl.layers.DenseLayer(x, n_units=self.hidden, name='core_fc1')
            x = tl.layers.ReshapeLayer(x, [-1, 1, self.hidden], name='core_input1_reshape')  # input should be [batch_size, n_steps, n_features]

            # fixme: tensorlayer 버그 많네;; RNNLayer initial_state parameter에 없다고 나옴
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
        theta = self.emission_net(lstm_2.outputs, mask, is_training, reuse=True)
        glimpse_vec, glimpse_patch = self.stgl_net(masked_image, theta, is_training, reuse=True)

        return lstm_1, lstm_2, glimpse_vec, glimpse_patch


# class SolutionNetwork(object):
#     def __init__(self, n_filters=[1024, 512, 128]):
#         self.n_filters = n_filters
#
#     def __call__(self, lstm_result, is_training, reuse):
#         """
#
#         :param lstm_result: tensor - self.hidden dimension [100]
#         :param is_training:
#         :return: layer
#         """
#         with tf.variable_scope("solution", reuse=reuse):
#             tl.layers.set_name_reuse(reuse)
#             print('Solution Network')
#             self.network = tl.layers.InputLayer(lstm_result, name='solution_input')
#             self.network = tl.layers.ReshapeLayer(self.network, [-1, 4, 4, 8], name='solution_reshape')
#
#             n_units = [1024, 512, 128]
#             for i in range(len(self.n_units)):
#                 idx = i + 1
#                 self.network = tl.layers.DenseLayer(self.network, n_units=n_units[i], name='solution_fc' + str(idx))
#                 self.network = tl.layers.BatchNormLayer(self.network, act=tf.nn.relu, is_train=is_training,
#                                                         name='solution_fc_bn' + str(idx))
#             self.network = tl.layers.ReshapeLayer(self.network, [-1, 4, 4, 8], name='solution_reshape')
#             solution_image = self.network
#
#         return solution_image

#### Decoder
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

            completed_image_network = tl.layers.InputLayer(z, name='completionn_input')  # dimension cram.hidden [100]
            completed_image_network = tl.layers.ReshapeLayer(completed_image_network, [-1, 1, 1, self.hidden],
                                                             name='completion_input_reshape')

            # deconvolution
            n_filters = 128
            idx = 1

            completed_image_size = completed_image_network.outputs.get_shape().as_list()[1]
            while completed_image_size < int(self.image_size//2):
                # 버그 있음 - tl.layers.DeConv2d n_filters 아니고 n_out_channel
                completed_image_network = tl.layers.DeConv2d(
                    completed_image_network, n_out_channel=n_filters, filter_size=(3, 3),
                    out_size=(int(completed_image_size*2), int(completed_image_size*2)),
                    strides=(2, 2), padding='SAME', name='complete_transconv' + str(idx))
                completed_image_network = tl.layers.BatchNormLayer(
                    completed_image_network, act=tf.nn.relu, is_train=is_training,
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
            # generated = tf.nn.relu(completed_image_network.outputs)
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
                net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_training,
                                               name='global_critic_conv_bn' + str(idx))
                # net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME',
                #                           name='global_critic_maxpool' + str(idx))

            net = tl.layers.FlattenLayer(net)
            net = tl.layers.DenseLayer(net, n_units=100)
            net = tl.layers.DenseLayer(net, n_units=1, name='global_critic_fc1')
            logits = net.outputs

        return tf.nn.sigmoid(logits), logits





