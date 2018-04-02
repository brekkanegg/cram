"""
ISSSUE
0. 상대경로
1. dataset
11

"""

# from model import ST_GlimpseNetwork, RecurrentNetwork, ContextNetwork, EmissionNetwork, SolutionNetwork, CRAM
from cram import CRAM

# todo
import tensorflow as tf
import tensorlayer as tl
import numpy as np

import sys, os
import random
import time

from cram import CRAM
from inputs import inputs

import scipy.misc


# parameters

flags = tf.app.flags
flags.DEFINE_bool("is_training", False, "train/test")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("epoch_c", 5, "Epoch to train considering reconstruction")


flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 8, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 256, "The size of image to use")
flags.DEFINE_string("data_dir", "coco/person256", "Dataset directory")

flags.DEFINE_integer("glimpse_num", 4, "number of glimpes")
flags.DEFINE_integer("glimpse_size", 32, "size of glimpes")
flags.DEFINE_integer("hidden", 10, "hidden dimension")
flags.DEFINE_float("mask_ratio", 0.25, "Mask size ratio")  # 되도록이면 바꾸지 말 것
flags.DEFINE_bool("use_wgan", False, "use w-gan")

flags.DEFINE_integer("max_to_keep", 5, "model number of max to keep")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_bool("checkpoint_override", True, "Overriding checkpoint")
flags.DEFINE_string("sample_dir", "sample", "save the image samples [samples]")
flags.DEFINE_string("summary_dir", "summary", "save the summary")

flags.DEFINE_string("gpu", "0", "# of gpu to use"),

flags.DEFINE_string("task", "inpainting", "inpainting/classification")
flags.DEFINE_string("model_name", 'model', 'later change')

FLAGS = flags.FLAGS



model_config = {'dataset': FLAGS.dataset,
                'glimpse_num': FLAGS.glimpse_num,
                'glimpse_size': FLAGS.glimpse_size,
                'hidden': FLAGS.hidden,
                'task': FLAGS.task
                }

model_dir = ['{}-{}'.format(key, model_config[key]) for key in sorted(model_config.keys())]
model_dir = '/'.join(model_dir)

# fixme: 상대경로
FLAGS.checkpoint_dir = os.path.join('Development/mk/cram', FLAGS.checkpoint_dir, model_dir)
FLAGS.sample_dir = os.path.join('Development/mk/cram', FLAGS.sample_dir, model_dir)
FLAGS.summary_dir = os.path.join('Development/mk/cram', FLAGS.summary_dir, model_dir)

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.summary_dir):
    os.makedirs(FLAGS.summary_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # load data
    test_inputs, test_init_op, test_counts = \
        inputs(FLAGS.data_dir, FLAGS.image_size, FLAGS.batch_size, 5, mode='test', central_crop=True)

    print('Test Data Counts: ', test_counts)
    sess.run(tf.local_variables_initializer())

    # sess.run(test_init_op)
    # a = sess.run(test_inputs)

    cram = CRAM(config=FLAGS, inputs=test_inputs)




    print('checkpoint_dir: ', FLAGS.checkpoint_dir)
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path and not FLAGS.checkpoint_override:  # retrain
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        saver.restore(sess, ckpt.model_checkpoint_path)

        graph = tf.get_default_graph()
        # todo: global step
        # tf.train.get_or_create_global_step(graph)

        print("""
======
An existing model was found in the checkpoint directory.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
======
        
        """)
    else:
        print("""
======
An existing model was not found in the checkpoint directory.
Initializing a new one.
======
        """)
        t1 = time.time()
        tl.layers.initialize_global_variables(sess)
        print('Initializing Time: ', time.time() - t1)


    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    #fixme: 프린트 줄이기

    # train
    print('Start testing')

    start_time = time.time()

    counter = 0
    for epoch in range(FLAGS.epoch):
        sess.run(test_init_op)
        batch_idxs = int(test_counts // FLAGS.batch_size)
        save_step = int(batch_idxs / 5)

        for idx in range(0, batch_idxs):
            try:
                if epoch < FLAGS.epoch_c:  # test for reconstruction first
                    summary, local_D_loss, global_D_loss, local_G_loss, global_G_loss, recon_loss, _ = sess.run(
                        [cram.all_summary, cram.local_D_loss, cram.global_D_loss, cram.local_G_loss, cram.global_G_loss, cram.recon_loss,
                         cram.recon_optim])
                    summary_writer.add_summary(summary, counter)

                else:
                    local_D_loss, global_D_loss, _ = sess.run([cram.local_D_loss, cram.global_D_loss, cram.d_optim])
                    summary, local_G_loss, global_G_loss, recon_loss, _ = sess.run([cram.all_summary, cram.local_G_loss, cram.global_G_loss, cram.recon_loss, cram.g_optim])
                    summary_writer.add_summary(summary, counter)


            except tf.errors.OutOfRangeError:
                sess.run(test_init_op)
                continue

            counter += 1

            print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, "
                  "local_D_loss: {:.8f}, global_D_loss: {:.8f}, local_G_loss: {:.8f}, global_G_loss: {:.8f}, "
                  "recon_loss: {:.8f}".format(
                epoch, idx, batch_idxs, time.time() - start_time,
                local_D_loss, global_D_loss, local_G_loss, global_G_loss, recon_loss))

            # if np.mod(counter, save_step) == save_step-1:
            if np.mod(counter, save_step) == 1:

                # todo: viz - orig_image, masked_image, completed_image // glimpse
                # save samples
                try:
                    orig_image, mask, masked_image, part_image = sess.run(
                        [cram.orig_image, cram.mask, cram.masked_image, cram.part_image])

                    completed_image, completed, generated = sess.run([cram.completed_image, cram.completed, cram.generated])
                    glimpses = sess.run(cram.glimpses)

                except tf.errors.OutOfRangeError:
                    sess.run(test_init_op)
                    continue

                _dir = os.path.join(FLAGS.sample_dir, str(counter))
                if not os.path.exists(_dir):
                    os.makedirs(_dir)

                # # temp - debug
                # for t in [part_image[0]]:
                #     scipy.misc.imsave(_dir + '/temp.jpg', t)

                # result visualization
                print('Visualization-')
                for im, mi, ci, i in zip(orig_image, masked_image, completed_image, range(len(orig_image))):
                    # blank = np.array()
                    c = np.concatenate([im, mi, ci], axis=1)
                    scipy.misc.imsave(_dir+'/result_'+str(i)+'.jpg', c)

                # glimpse visualization
                for i in range(len(glimpses)):
                    for gl, id in zip(glimpses[i], range(len(glimpses[i]))):
                        scipy.misc.imsave(_dir + '/glimpse_{}_{}.jpg'.format(id, i), gl)

                # for a, b, c, d, i in zip(orig_image, mask, masked_image, completed_image, range(len(orig_image))):
                #     # blank = np.array()
                #     x = np.concatenate([a, b*255, c, d], axis=1)
                #     scipy.misc.imsave(_dir+'/result_'+str(i)+'.jpg', x)
                #
                # for a, b, c, i in zip(completed, part_image, generated, range(len(orig_image))):
                #     # blank = np.array()
                #     x = np.concatenate([a, b, c], axis=1)
                #     scipy.misc.imsave(_dir+'/result_part'+str(i)+'.jpg', x)

                tl.files.save_ckpt(sess, mode_name=cram.model_name,
                                   save_dir=FLAGS.checkpoint_dir, printable=False)

