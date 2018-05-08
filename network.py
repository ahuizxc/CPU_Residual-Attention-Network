import numpy as np
import tensorflow as tf

from math import ceil

class Network(object):
    """
    Network class, includes implementation of cnn layers.
    """
    def __init__(self, inputs, is_train=True, trainable=True):
        """
        Initialization
        """
        self.inputs = inputs
        self.is_train = is_train
        self.trainable = trainable
        self.setup()

    def setup(self):
        """
        This function defines the cnn model,
        need to be implemented in subclasses
        """
        raise NotImplementedError("Need to be implemented in subclasses")

    @staticmethod
    def make_cpu_variables(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
        return var

    def conv(self,
             x,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu,
             group=1,
             bias_term=False,
             padding="SAME",
             trainable=True):
        """
        Function for convolutional layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- k_h: Height of kernels
        --- k_w: Width of kernels
        --- c_o: Amount of kernels
        --- s_h: Stride in height
        --- s_w: Stride in width
        --- name: Layer name
        --- relu: Do relu or not
        --- group: Amount of groups
        --- bias_term: Add bias or not
        --- padding: Padding method, SAME or VALID
        --- trainable: Whether the parameters in this layer are trainable
        Output:
        --- outputs: Output of the convolutional layer
        """
        with tf.name_scope(name), tf.variable_scope(name):
            # Get the input channel
            c_i = x.get_shape()[-1]/group
            # Create the weights, with shape [k_h, k_w, c_i, c_o]
            weights = self.make_cpu_variables("weights", [k_h, k_w, c_i, c_o], trainable=trainable)
            # Create a function for convolution calculation
            def conv2d(i, w):
                return tf.nn.conv2d(i, w, [1, s_h, s_w, 1], padding)
            # If we don't need to divide this convolutional layer
            if group == 1:
                outputs = conv2d(x, weights)
            # If we need to divide this convolutional layer
            else:
                # Split the input and weights
                group_inputs = tf.split(x, group, 3, name="split_inputs")
                group_weights = tf.split(weights, group, 3, name="split_weights")
                group_outputs = [conv2d(i, w) for i, w in zip(group_inputs, group_weights)]
                # Concatenate the groups
                outputs = tf.concat(group_outputs, 3)
            if bias_term:
                # Create the biases, with shape [c_o]
                biases = self.make_cpu_variables("biases", [c_o], trainable=trainable)
                # Add the biases
                outputs = tf.nn.bias_add(outputs, biases)
            if relu:
                # Nonlinear process
                outputs = tf.nn.relu(outputs)
            # Return layer's output
            return outputs

    @staticmethod
    def max_pool(x,
                 k_h,
                 k_w,
                 s_h,
                 s_w,
                 name,
                 padding="VALID"):
        """
        Function for max pooling layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- k_h: Height of kernels
        --- k_w: Width of kernels
        --- s_h: Stride in height
        --- s_w: Stride in width
        --- name: Layer name
        --- padding: Padding method, SAME or VALID
        Output:
        --- outputs: Output of the max pooling layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.max_pool(x, [1, k_h, k_w, 1], [1, s_h, s_w, 1], padding)
            # Return layer's output
            return outputs

    @staticmethod
    def avg_pool(x,
                 k_h,
                 k_w,
                 s_h,
                 s_w,
                 name,
                 padding="VALID"):
        """
        Function for average pooling layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- k_h: Height of kernels
        --- k_w: Width of kernels
        --- s_h: Stride in height
        --- s_w: Stride in width
        --- name: Layer name
        --- padding: Padding method, SAME or VALID
        Output:
        --- outputs: Output of the average pooling layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.avg_pool(x, [1, k_h, k_w, 1], [1, s_h, s_w, 1], padding)
            # Return layer's output
            return outputs

    @staticmethod
    def relu(x, name):
        """
        Function for relu layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- name: Layer name
        Output:
        --- outputs: Output of the relu layer
        """

        with tf.name_scope(name):
            outputs = tf.nn.relu(x)
            # Return layer's output
            return outputs

    def fc(self, x, nout, name, relu, bias_term=False, trainable=True):
        """
        Function for fully connected layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- nout: Amount of output neurals
        --- name: Layer name
        --- relu: Do relu or not
        --- trainable: Whether the parameters in this layer are trainable
        Output:
        --- outputs: Output of the fc layer
        """
        with tf.name_scope(name), tf.variable_scope(name):
            # Reshape the input
            input_shape = x.get_shape()
            # If the input is 4-D, reshape it to 2-D
            if len(input_shape) == 4:
                dim = 1
                for d in input_shape.as_list()[1:]:
                    dim *= d
                x = tf.reshape(x, [-1, dim])
            else:
                dim = input_shape.as_list()[1]
            # Get the weights, with shape [dim, nout]
            weights = self.make_cpu_variables("weights", [dim, nout], initializer=tf.truncated_normal_initializer(stddev=0.001), trainable=trainable)
            # Matmul
            outputs = tf.matmul(x, weights)
            if bias_term:
                # Get the biases, with shape [nout]
                biases = self.make_cpu_variables("biases", [nout], trainable=trainable)
                # Add the biases
                outputs = tf.nn.bias_add(outputs, biases)
            if relu:
                outputs = tf.nn.relu(outputs)
            # Return layer's output
            return outputs

    @staticmethod
    def dropout(x, keep_prob, name):
        """
        Function for dropout layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- keep_prob: Keep probability for dropout layer
        --- name: Layer name
        Output:
        --- outputs: Output of the dropout layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.dropout(x, keep_prob)
            # Return layer's output
            return outputs

    @staticmethod
    def upsample(x, name, size):
        """
        Function for upsample layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- name: Layer name
        --- size: Upsample size
        Output:
        --- outputs: Output of the upsample layer
        """
        with tf.name_scope(name):
            outputs = tf.image.resize_bilinear(x, size)
            # Return layer's output
            return outputs

    @staticmethod
    def softmax(x, name):
        """
        Function for softmax layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- name: Layer name
        Output:
        --- outputs: Output of the softmax layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.softmax (x)
            # Return layer's output
            return outputs

    @staticmethod
    def batch_normal(x, is_train, name, activation_fn=None):
        """
        Function for batch normalization

        Input:
        --- x: input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- is_train: Is training or not
        --- name: Layer name
        --- activation_fn: Activation function
        Output:
        --- outputs: Output of batch normalization
        """
        with tf.name_scope(name), tf.variable_scope(name):
            outputs = tf.contrib.layers.batch_norm(x,
                                                   decay=0.999,
                                                   scale=True,
                                                   activation_fn=activation_fn,
                                                   is_training=is_train)
            return outputs
