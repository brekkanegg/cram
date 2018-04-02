"""
ISSSUE
0. 상대경로
1. dataset
11

"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np

import sys, os
import time

from cram import CRAM
from inputs import *

import scipy.misc
import pprint


# Parameters
flags = tf.app.flags
flags.DEFINE_string("task", "inpainting", "inpainting/classification")
flags.DEFINE_bool("is_training", True, "train/test")
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_integer("epoch_c", 0, "Epoch to train considering reconstruction")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

flags.DEFINE_string("data_dir", "svhn", "Dataset directory: coco/person256, places365")
flags.DEFINE_integer("image_size", 32, "The size of image to use")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")

flags.DEFINE_integer("glimpse_num", 4, "number of glimpes")
flags.DEFINE_integer("glimpse_size", 16, "size of glimpes")
flags.DEFINE_integer("hidden", 512, "hidden dimension")
flags.DEFINE_float("mask_ratio", 0.25, "Mask size ratio")  # 되도록이면 바꾸지 말 것
flags.DEFINE_bool("use_wgan", False, "use w-gan")

flags.DEFINE_integer("max_to_keep", 5, "model number of max to keep")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_bool("override", False, "Overriding checkpoint")
flags.DEFINE_string("sample_dir", "sample", "save the image samples [samples]")
flags.DEFINE_string("summary_dir", "summary", "save the summary")

flags.DEFINE_integer("print_step", 100, "printing interval")
flags.DEFINE_integer("save_step", 500, "saving_interval")


flags.DEFINE_string("gpu", "0", "# of gpu to use"),

FLAGS = flags.FLAGS


model_config = {'task': FLAGS.task,
                'data_dir': FLAGS.data_dir,
                'glimpse_num': FLAGS.glimpse_num,
                'glimpse_size': FLAGS.glimpse_size,
                'hidden': FLAGS.hidden
                }

model_dir = ['{}-{}'.format(key, model_config[key]) for key in sorted(model_config.keys())]
model_dir = '/'.join(model_dir)
print('CONFIG: ')
pprint.pprint(model_config)

# fixme: 상대경로
FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
FLAGS.sample_dir = os.path.join(FLAGS.sample_dir, model_dir)
FLAGS.summary_dir = os.path.join(FLAGS.summary_dir, model_dir)
#
# FLAGS.checkpoint_dir = os.path.join('Development/mk/cram', FLAGS.checkpoint_dir, model_dir)
# FLAGS.sample_dir = os.path.join('Development/mk/cram', FLAGS.sample_dir, model_dir)
# FLAGS.summary_dir = os.path.join('Development/mk/cram', FLAGS.summary_dir, model_dir)

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

    # Load Data
    if FLAGS.data_dir == 'svhn':
        inputs_func = inputs_svhn
    else:  # places365, coco/person256
        inputs_func = inputs_big

    train_inputs, train_init_op, train_counts = \
        inputs_func(FLAGS.data_dir, FLAGS.image_size, FLAGS.batch_size, 5, mode='train', central_crop=True)

    print('Train Data Counts: ', train_counts)
    sess.run(tf.local_variables_initializer())

    # sess.run(train_init_op)
    # s = sess.run(train_inputs)
    #
    # Model
    cram = CRAM(config=FLAGS, inputs=train_inputs)

    # Try Loading Checkpoint
    print('Checkpoint: ', FLAGS.checkpoint_dir)
    print(" [*] Reading Checkpoint...")
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path and not FLAGS.override:  # retrain
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        t1 = time.time()
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restoring Time: ', time.time() - t1)

        # graph = tf.get_default_graph()
        # todo: global step
        # tf.train.get_or_create_global_step(graph)

        print("""
======
An existing model was found in the checkpoint directory.
Loading finished.
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

    # Train
    print('Start Training')

    start_time = time.time()

    counter = 0
    epoch = 0

    # fixme: feed_dict
    for epoch in range(FLAGS.epoch):
        sess.run(train_init_op)
        batch_idxs = int(train_counts // FLAGS.batch_size)

        save_step = FLAGS.save_step
        print_step = FLAGS.print_step

        g = sess.run(cram.glimpses)

        for idx in range(0, batch_idxs):
            try:
                if epoch < FLAGS.epoch_c:  # train for reconstruction first
                    z, local_D_loss, global_D_loss, local_G_loss, global_G_loss, recon_loss, _ = sess.run(
                        [cram.z, cram.local_D_loss, cram.global_D_loss, cram.local_G_loss, cram.global_G_loss,
                         cram.recon_loss, cram.recon_optim])
                    # summary_writer.add_summary(summary, counter)
                    counter += 1

                else:
                    local_D_loss, global_D_loss, _ = sess.run([cram.local_D_loss, cram.global_D_loss, cram.d_optim])
                    local_G_loss, global_G_loss, recon_loss, _ = sess.run(
                        [cram.local_G_loss, cram.global_G_loss, cram.recon_loss, cram.g_optim])
                    # summary_writer.add_summary(summary, counter)
                    counter += 2

            except tf.errors.OutOfRangeError:
                break

    
            if np.mod(idx, print_step) == 1:
                # print(z[0])
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, "
                      "local_D_loss: {:.8f}, global_D_loss: {:.8f}, local_G_loss: {:.8f}, global_G_loss: {:.8f}, "
                      "recon_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time,
                    local_D_loss, global_D_loss, local_G_loss, global_G_loss, recon_loss))

            # Save model and sample image files / summary as well
            if np.mod(counter, save_step) == 1 or np.mod(counter, save_step) == 2:

                # Save Samples
                try:
                    summary, orig_image, mask, masked_image, part_image, \
                    completed_image, completed, generated, glimpses = sess.run(
                        [cram.all_summary, cram.orig_image, cram.mask, cram.masked_image, cram.part_image,
                         cram.completed_image, cram.completed, cram.generated, cram.glimpses])
                    summary_writer.add_summary(summary, counter)
                    counter += 1

                except tf.errors.OutOfRangeError:
                    break


                # # temp - debug
                # for t in [part_image[0]]:
                #     scipy.misc.imsave(_dir + '/temp.jpg', t)

                # Visualization
                viz_dir = os.path.join(FLAGS.sample_dir, str(counter))
                if not os.path.exists(viz_dir):
                    os.makedirs(viz_dir)

                # result visualization
                print('[$] Visualization- at ', viz_dir)
                for im, mi, ci, gi, i in \
                        zip(orig_image, masked_image, completed_image, generated, range(len(orig_image))):
                    # blank = np.array()
                    c = np.concatenate([im, mi, ci, gi], axis=1)
                    scipy.misc.imsave(viz_dir + '/result_'+str(i)+'.jpg', c)

                # glimpse visualization
                for i in range(len(glimpses)):
                    for gl, iid in zip(glimpses[i], range(len(glimpses[i]))):
                        scipy.misc.imsave(viz_dir + '/glimpse_{}_{}.jpg'.format(iid, i), gl)

                tl.files.save_ckpt(sess, mode_name=cram.model_name, save_dir=FLAGS.checkpoint_dir, printable=False)

