"""

"""

import tensorflow as tf
import os, sys
import numpy as np
import time
import pprint

from vgg import VGG
from sram import SRAM
import inputs

# Parameters
flags = tf.app.flags
flags.DEFINE_bool("is_training", True, "train/test")
flags.DEFINE_bool("use_gt", False, "use ground truth segmentation")

flags.DEFINE_string("dataset", "cifar10", "willow, cub")
flags.DEFINE_integer("option", 0, "option number")

flags.DEFINE_bool("sal", False, "false")
flags.DEFINE_bool("x255", False, "false")


flags.DEFINE_integer("gn", 6, "number of glimpes")
flags.DEFINE_integer("gs", 8, "size of glimpes")
flags.DEFINE_integer("h", 256, "hidden dimension")

flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("lr", 1e-4, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")

flags.DEFINE_integer("bs", 64, "The size of batch images [32]")

flags.DEFINE_integer("max_to_keep", 5, "model number of max to keep")
flags.DEFINE_bool("ov", False, "Overriding checkpoint")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "save the image samples [samples]")
flags.DEFINE_string("summary_dir", "summary", "save the summary")

flags.DEFINE_integer("print_step", 100, "printing interval")
flags.DEFINE_integer("save_step", 1000, "saving_interval")

flags.DEFINE_string("gpu", "1", "# of gpu to use"),

FLAGS = flags.FLAGS

model_config = {'saliency': FLAGS.sal,
                'x255': FLAGS.x255,
                'glimpse_num': FLAGS.gn,
                'glimpse_size': FLAGS.gs,
                'hidden': FLAGS.h,
                'dataset': FLAGS.dataset,
                'learning_rate': FLAGS.lr,
                'gt_seg': FLAGS.use_gt,
                'option': FLAGS.option,
                }

model_dir = ['{}-{}'.format(key, model_config[key]) for key in sorted(model_config.keys())]
model_dir = '/'.join(model_dir)
print('CONFIG: ')
pprint.pprint(model_config)
print('Override: ', FLAGS.ov)
FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
FLAGS.sample_dir = os.path.join(FLAGS.sample_dir, model_dir)
FLAGS.summary_dir = os.path.join(FLAGS.summary_dir, model_dir)

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.summary_dir):
    os.makedirs(FLAGS.summary_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

###

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    
    if FLAGS.dataset == 'cifar10':
        _dl = inputs.dataloader_cifar10
    elif FLAGS.dataset == 'cifar100':
        _dl = inputs.dataloader_cifar100
    # elif FLAGS.dataset == 'imagenet':
    #     _dl = inputs.dataloader_imagenet
    elif FLAGS.dataset == 'cub200':
        _dl = inputs.dataloader_cub200

    # if FLAGS.is_control:
    #     train_inputs = _dl(FLAGS.bs, saliency=True, mode='control')

    if FLAGS.use_gt:
        train_inputs = _dl(FLAGS.bs, saliency=True, mode='gt', x255=FLAGS.x255)
        val_inputs = _dl(FLAGS.bs, saliency=True, mode='val', reuse=False, x255=FLAGS.x255)

    else:
        train_inputs = _dl(FLAGS.bs, saliency=FLAGS.sal, mode='train', reuse=False, x255=FLAGS.x255)
        val_inputs = _dl(FLAGS.bs, saliency=FLAGS.sal, mode='val', reuse=True, x255=FLAGS.x255)

    print('Train Data Counts: ', train_inputs.data_count)
    
    # Model
    model = SRAM(config=FLAGS, inputs=train_inputs)
    sess.run(tf.local_variables_initializer())

    # Try Loading Checkpoint
    print('Checkpoint: ', FLAGS.checkpoint_dir)
    print(" [*] Reading Checkpoint...")
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path and not FLAGS.ov:  # retrain
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
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    # Train

    print('CONFIG: ')
    pprint.pprint(model_config)
    print('\nStart Training')

    start_time = time.time()

    counter = 0
    epoch = 0

    save_step = FLAGS.save_step
    print_step = FLAGS.print_step

    # fixme: feed_dict
    min_val_loss = np.inf
    max_val_acc = 0
    stop_stack = 0

    batch_idxs = int(train_inputs.data_count // FLAGS.bs)
    for epoch in range(FLAGS.epoch):
        # if np.mod(epoch, 10) == 1 and epoch != 1:
        #     FLAGS.learning_rate *= 0.3
        #     print('Learning Rate Decreased to: ', FLAGS.learning_rate)

        if epoch == 100:
            FLAGS.lr *= 1e-1
            print('Learning Rate Decreased to: ', FLAGS.lr)
        if epoch == 200:
            FLAGS.lr *= 1e-1
            print('Learning Rate Decreased to: ', FLAGS.lr)
        if epoch == 300:
            FLAGS.lr *= 1e-1
            print('Learning Rate Decreased to: ', FLAGS.lr)



        train_inputs.shuffle()  # shuffle
        for idx in range(0, batch_idxs):
            try:
                batch_xs, batch_ys = train_inputs.next_batch()
            except ValueError: #, FileNotFoundError):
                train_inputs.pointer += 1
                batch_xs, batch_ys = train_inputs.next_batch()


            ##### debugging

            # a, b, c, d = sess.run([model.cross_entropy_loss, model.clue_glimpse_loss, model.logits, model.y],
            #                       feed_dict={model.x: batch_xs, model.y: batch_ys, model.clue: batch_clues})
            #####

            l, a, s, _ = sess.run([model.cross_entropy_loss, model.accuracy, model.summary_merge, model.train_op],
                                  feed_dict={model.x: batch_xs, model.y: batch_ys, model.is_training: True,
                                             model.learning_rate: FLAGS.lr})
            counter += 1

            # print
            if np.mod(counter, print_step) == 1:
                print("Epoch: [{:2d}] [{:4d}/{:4d}] [{:4d}] time: {:.4f}, "
                      "accuracy:  {:.4f} cls_loss: {:.6f}".format(
                    epoch, idx, batch_idxs, counter-1, time.time() - start_time, a, l))


            # Save model and sample image files / summary as well
            if np.mod(counter, save_step) == 1 and counter is not 1:
                summary_writer.add_summary(s, counter)

                # validation

                cls_loss = 0
                predictions = []
                labels = []
                val_batch_idxs = int(val_inputs.data_count // FLAGS.bs)
                for vi in range(0, val_batch_idxs):

                    # fixme
                    try:
                        bv_xs, bv_ys = val_inputs.next_batch()
                    except ValueError:
                        val_inputs.pointer += 1
                        bv_xs, bv_ys = val_inputs.next_batch()

                    cl, lg, lb = sess.run([model.cross_entropy_loss, model.logits, model.y],
                                          feed_dict={model.x: bv_xs, model.y: bv_ys, model.is_training: False})

                    cls_loss += cl
                    predictions.extend(np.argmax(lg, 1))
                    labels.extend(lb)

                val_loss = cls_loss
                val_loss_sum = tf.Summary(value=[
                    tf.Summary.Value(tag="val_loss", simple_value=val_loss),
                ])
                summary_writer.add_summary(val_loss_sum, counter)

                val_acc = sum(np.equal(predictions, labels)) / len(labels)
                val_acc_sum = tf.Summary(value=[
                    tf.Summary.Value(tag="val_acc", simple_value=val_acc),
                ])
                summary_writer.add_summary(val_acc_sum, counter)
                print("Validation Accuracy: {:.4f}".format(val_acc))
                print('Max Val Accuracy: {:.4f}'.format(max_val_acc))

                # if cls_loss < min_val_loss:
                if val_acc > max_val_acc:
                    min_val_loss = cls_loss
                    max_val_acc = val_acc
                    saver.save(sess, FLAGS.checkpoint_dir + '/{}.ckpt'.format(model.model_name))
                    print('Model saved at: {}/{}.ckpt'.format(FLAGS.checkpoint_dir, model.model_name))
                    print('Max Val Accuracy: {:.4f}'.format(max_val_acc))
                    pprint.pprint(model_config)
                # else:
                #     if a > max_val_acc + 0.5: ## overfitting
                #         sys.exit("Stop Training! Max Val Accuracy: {} Iteration: {} Time Spent: {:.4f}"
                #                  .format(max_val_acc, counter, time.time() - start_time))

                    # stop_stack += 1
                    # print('Stop Stack: {}/100, Iterations: {}'.format(stop_stack, counter))
                    # # model.learning_rate *= 0.1  # learning rate *10
                    # if stop_stack == 100:
                    #     sys.exit("Stop Training! Iteration: {} Time Spent: {}".format(counter, time.time() - start_time))


    print('Training finished')
    pprint.pprint(model_config)