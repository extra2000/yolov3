import os
import time
import logging

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from . import YOLOv3
from .dataset_generator import DatasetGenerator
from common.read_class_names import read_class_names

logger = logging.getLogger(__name__)
tf.compat.v1.disable_eager_execution()


class YOLOTrain(object):
    """YOLOv3 Training
    """

    def __init__(self, config, initial_weight, logdir, output_weight_dir):
        """Implements YOLOv3 training

        Parameters
        ----------
        config : dict
            Configurations.
        initial_weight : str
            Initial weight filename.
        logdir : str
            Directory to store training logs.
        output_weight_dir : str
            Directory to store trained weights.
        """

        if os.path.exists(logdir):
            raise FileExistsError(logdir)
        if os.path.exists(output_weight_dir):
            raise FileExistsError(output_weight_dir)

        logger.info('Creating "{}"'.format(logdir))
        os.makedirs(logdir)

        logger.info('Creating "{}"'.format(output_weight_dir))
        os.makedirs(output_weight_dir)

        # Sometimes, GPU training may get stuck on self.sess.run() for 30-20 minutes.
        # Thus, using session_timeout will exit current self.sess.run() and retry again.
        self.session_timeout = 5000  # ms

        self.output_weight_dir   = output_weight_dir
        self.anchor_per_scale    = config['yolov3']['anchor_per_scale']
        self.classes             = read_class_names(config['yolov3']['classnames'])
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = float(config['yolov3']['training']['learn_rate_init'])
        self.learn_rate_end      = float(config['yolov3']['training']['learn_rate_end'])
        self.first_stage_epochs  = config['yolov3']['training']['first_stage_epochs']
        self.second_stage_epochs = config['yolov3']['training']['second_stage_epochs']
        self.warmup_periods      = config['yolov3']['training']['warmup_epochs']
        self.initial_weight      = initial_weight
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = config['yolov3']['moving_ave_decay']
        self.max_bbox_per_scale  = 150
        self.trainset            = DatasetGenerator(config, 'train')
        self.testset             = DatasetGenerator(config, 'test')
        self.steps_per_period    = len(self.trainset)
        self.sess                = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))

        with tf.name_scope('define_input'):
            self.input_data   = tf.compat.v1.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.compat.v1.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.compat.v1.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.compat.v1.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable     = tf.compat.v1.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = YOLOv3(self.input_data, self.trainable, config)
            self.net_var = tf.compat.v1.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.compat.v1.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.compat.v1.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.compat.v1.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.compat.v1.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.compat.v1.trainable_variables()
            second_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.compat.v1.train.Saver(self.net_var)
            self.saver  = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.compat.v1.summary.scalar("learn_rate",      self.learn_rate)
            tf.compat.v1.summary.scalar("giou_loss",  self.giou_loss)
            tf.compat.v1.summary.scalar("conf_loss",  self.conf_loss)
            tf.compat.v1.summary.scalar("prob_loss",  self.prob_loss)
            tf.compat.v1.summary.scalar("total_loss", self.loss)

            self.write_op = tf.compat.v1.summary.merge_all()
            self.summary_writer  = tf.compat.v1.summary.FileWriter(logdir, graph=self.sess.graph)


    def train(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        try:
            logger.info('Restoring weights from: "{}"'.format(self.initial_weight))
            self.loader.restore(self.sess, self.initial_weight)
        except:
            logger.fatal('Cannot load "{}". Otherwise, YOLOv3 training will start from scratch'.format(self.initial_weight))
            self.first_stage_epochs = 0
            raise Exception

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                while True:
                    try:
                        _, summary, train_step_loss, global_step_val = self.sess.run(
                            [train_op, self.write_op, self.loss, self.global_step],
                            feed_dict={
                                self.input_data:   train_data[0],
                                self.label_sbbox:  train_data[1],
                                self.label_mbbox:  train_data[2],
                                self.label_lbbox:  train_data[3],
                                self.true_sbboxes: train_data[4],
                                self.true_mbboxes: train_data[5],
                                self.true_lbboxes: train_data[6],
                                self.trainable:    True,
                            },
                            options=tf.compat.v1.RunOptions(timeout_in_ms=self.session_timeout)
                        )

                        train_epoch_loss.append(train_step_loss)
                        self.summary_writer.add_summary(summary, global_step_val)
                        pbar.set_description("train loss: {:.2f}".format(train_step_loss))

                    except tf.errors.DeadlineExceededError as e:
                        logger.warning(e)
                        logger.warning("Retrying ...")
                    else:
                        break

            pbartest = tqdm(self.testset)

            for test_data in pbartest:
                while True:
                    try:
                        test_step_loss = self.sess.run(
                            self.loss,
                            feed_dict={
                                self.input_data:   test_data[0],
                                self.label_sbbox:  test_data[1],
                                self.label_mbbox:  test_data[2],
                                self.label_lbbox:  test_data[3],
                                self.true_sbboxes: test_data[4],
                                self.true_mbboxes: test_data[5],
                                self.true_lbboxes: test_data[6],
                                self.trainable:    False,
                            },
                            options=tf.compat.v1.RunOptions(timeout_in_ms=self.session_timeout)
                        )

                        test_epoch_loss.append(test_step_loss)
                        pbartest.set_description("test loss: {:.2f}".format(test_step_loss))

                    except tf.errors.DeadlineExceededError as e:
                        logger.warning(e)
                        logger.warning("Retrying ...")
                    else:
                        break

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = os.path.join(self.output_weight_dir, "yolov3_test_loss={:.4f}.ckpt".format(test_epoch_loss))
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            logger.info('Epoch: {} ; Time: {} ; Train loss: {:.4f} ; Test loss: {:.4f}'.format(epoch, log_time, train_epoch_loss, test_epoch_loss))

            logger.info('Saving "{}"'.format(ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)
            logger.info('Successfully saved "{}"'.format(ckpt_file))
