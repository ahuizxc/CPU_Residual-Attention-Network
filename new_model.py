#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:42:54 2018

@author: liushenghui
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import pdb
from model import ResAttentionNet
from utils import inv_preprocess
from keras.utils import to_categorical
import cPickle
# Default values
MODE = "train"
RESTORE = False

# Configures 
bsize = 32
nclass = 10
num_gpus = 2
lr_base = 0.1
momentum = 0.9
weight_decay = 0.0001
average_decay = 0.999
steps_per_epoch = 1000
training_epochs = 1800
image_size = (32, 32)

# Dirs
ckpt_dir = "./ckpts"
data_list_dir = "./"

class ResAttentionNetClassification(object):
    """
    ResAttentionNetClassification class definition
    """
    def __init__(self, param_path="DEFAULT"):
        # Get the arguments
        self.args = self.get_arguments()
        self.load_img()
    def load_img(self):
        data = []
        label = []
        files = ['cifar10/data_batch_1','cifar10/data_batch_2','cifar10/data_batch_3','cifar10/data_batch_4','cifar10/data_batch_5']
        for file in files:
            with open(file,'rb') as f:
                dataset = cPickle.load(f)
                data.append(np.array(dataset['data'])/255)
                label += dataset['labels']
        data = np.array(data)
        self.data = data.reshape((-1,32,32,3))
        self.label = label
    def save(self, saver, sess, global_step):
        """
        Function for saving the model
        """
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
        saver.save(sess, ckpt_path, global_step)

    def load(self, loader, sess):
        """
        Function for loading the model
        """
        if os.path.exists(ckpt_dir):
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                loader.restore(sess, ckpt.model_checkpoint_path)

    @staticmethod
    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train(self):
        """
        Function to train the model
        """
        with tf.Graph().as_default(), tf.device("/cpu:0"):
            with tf.name_scope("optimizer"):
                global_step = tf.Variable(0, trainable=False, name="global_step")
                # Get the learning rate
                lr = tf.placeholder(dtype=tf.float32, name="learning_rate")
                # Create the optimizer objects
                optimizer = tf.train.MomentumOptimizer(lr, momentum, use_nesterov=True)
            # Create a reader object for train
            tower_grads_vals = []
            with tf.variable_scope(tf.get_variable_scope()):
                    # Dequeue one batch for the GPU
                train_img_bat = tf.placeholder(shape=(None,32,32,3),dtype=tf.float32)
                train_lab_bat = tf.placeholder(shape=(None),dtype=tf.int32)
                train_net = ResAttentionNet(train_img_bat)
                # --Inference:
                # Create the Residual Attention Network
                
                # Get the output of the network
                train_raw_preds = train_net.raw_score
                # --Define the loss function:
                # Get the weights of layers
                weight_list = [w for w in tf.trainable_variables() if "weights" in w.name]
                with tf.name_scope("loss"):
                    with tf.name_scope("reg_loss"):
                        # Get the reg loss
                        reg_loss = [weight_decay * tf.nn.l2_loss(w) for w in weight_list]
                        reg_loss = tf.add_n(reg_loss, "reg_loss")
                    with tf.name_scope("data_loss"):
                        # Get the data loss
                        data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_lab_bat, logits=train_raw_preds)
                        data_loss = tf.reduce_mean(data_loss, name="data_loss")
                    # Get the total loss
                    loss = tf.add(data_loss, reg_loss, "total_loss")
                my_train_op = optimizer.minimize(loss)
                # Reuse variables for the next tower
                tf.get_variable_scope().reuse_variables()
                # Get the trainable variables
                vals = tf.trainable_variables()
                # Get the gradients
                grads = tf.gradients(loss, vals)
                grads_vals = zip(grads, vals)
                tower_grads_vals.append(grads_vals)
            # Get moving average for losses
            loss_average = tf.train.ExponentialMovingAverage(0.9, name='avg')
            loss_average_op = loss_average.apply([loss, reg_loss, data_loss])
            grads_vals = self.average_gradients(tower_grads_vals)
            with tf.name_scope("optimizer"):
                # Define the optimizer:
                # Track the moving averages of all trainable variables
                ema = tf.train.ExponentialMovingAverage(average_decay, global_step)
                ema_op = ema.apply(tf.trainable_variables())
                # training op
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):            
                    update_op = optimizer.apply_gradients(grads_vals, global_step)
                with tf.control_dependencies([update_op, tf.group(loss_average_op)]):
                    train_op = tf.group(ema_op)
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope("evaluate"):
                    # Create a reader object for test
                    with tf.name_scope("create_test_inputs"):
                        # Dequeue one batch
                        test_img_bat = tf.placeholder(shape=(None,32,32,3),dtype=tf.float32)
                        test_lab_bat = tf.placeholder(shape=(None),dtype=tf.int32)
                        # Reuse variables
                        tf.get_variable_scope().reuse_variables()
                        # --Inference:
                        # Create the Residual Attention Network
                        test_net = ResAttentionNet(test_img_bat)
                        # Get the output of the network
                        test_raw_preds = test_net.raw_score
                        # Get the prediction
                        test_preds = tf.argmax(test_raw_preds, axis=1)
                        # Metrics
                        with tf.name_scope("metrics"):
                            # Accuracy
                            eval_accu = tf.reduce_mean(tf.cast(tf.equal(test_lab_bat, tf.cast(test_preds,dtype=tf.int32)),dtype=tf.float32))
            # Summary
            with tf.name_scope("summary"):
                # loss summary
                tf.summary.scalar("loss_raw", loss)
                tf.summary.scalar("reg_loss_raw", reg_loss)
                tf.summary.scalar("data_loss_raw", data_loss)
                tf.summary.scalar("loss_ave", loss_average.average(loss))
                tf.summary.scalar("reg_loss_ave", loss_average.average(reg_loss))
                tf.summary.scalar("data_loss_ave", loss_average.average(data_loss))
                # learning rate summary
#                tf.summary.scalar("learning_rate", lr)                
                # accuracy summary
#                tf.summary.scalar("eval_acc", eval_accu)
                # Image summary.
                images_summary = tf.py_func(inv_preprocess, [train_img_bat, 2], tf.uint8)
                tf.summary.image('images', images_summary, max_outputs=2)
                # Merge
                merge = tf.summary.merge_all()
            # Create Saver objects for save and restore
            saver = tf.train.Saver(max_to_keep=10)
            # Create a initializer
            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            # GPU config 
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                # Initialize the variables
                sess.run([init_g, init_l])
                # Open the Tensorboard
                writer = tf.summary.FileWriter("./graphs", graph=sess.graph)
                # Start queue threads.
                # Train the model
                print "{} -- Start Training:".format(datetime.now())
                gstep = global_step.eval(session=sess)
                training_epochs = 100
                batch_size = 4
                lr_value = 0.03
                train_x = self.data[:45000,:,:,:]
                train_y = self.label[:45000]
                test_x = self.data[45000:,:,:,:]
                test_y = self.label[45000:]
                for epoch in range(training_epochs):
                    cnt = 0
                    while cnt*batch_size+batch_size < train_x.shape[0]:
                        img = train_x[cnt*batch_size:cnt*batch_size+batch_size,:,:,:]
                        label = train_y[cnt*batch_size:cnt*batch_size+batch_size]
                        lo, m, _ =  sess.run([loss, merge, my_train_op], feed_dict={train_img_bat:img, train_lab_bat: label,lr: lr_value})
                        if cnt%100 == 0:
                            print "{0}: Epoch {1} Step {2}, the training loss = {3}".format(datetime.now(), epoch + 1,cnt, lo)
                        cnt += 1
                    eval_acc = sess.run([eval_accu], feed_dict={test_img_bat:test_x,test_lab_bat:test_y})
                    print 'Epoch: '+ str(epoch+1)+', testing accuracy: '+str(eval_acc)
            writer.close()
    @staticmethod
    def get_arguments():
        # Create a parser object
        parser = argparse.ArgumentParser()
        # Add arguments
        parser.add_argument("--mode", default=MODE, type=str, help="Mode: train, eval, infer")
        parser.add_argument("--restore", default=RESTORE, type=bool, help="Whether to restore")
        # Parse the arguments
        return parser.parse_args()


if __name__ == "__main__":
    model = ResAttentionNetClassification()
    model.train()
