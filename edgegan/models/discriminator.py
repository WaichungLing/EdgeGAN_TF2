import tensorflow as tf
from edgegan import nn


class Discriminator(object):
    def __init__(self, name, is_train, norm='batch', activation='lrelu',
                 num_filters=64, use_resnet=False):
        print(' [*] Init Discriminator %s', name)
        self._num_filters = num_filters
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._use_resnet = use_resnet
        self._reuse = False

    def __call__(self, input, reuse=False):
        if self._use_resnet:
            return self._resnet(input)
        else:
            return self._convnet(input)

    def _resnet(self, input):
        # return None
        with tf.compat.v1.variable_scope(self.name, reuse=self._reuse):
            D = nn.residual2(input, self._num_filters, 'd_resnet_0', 3, 1,
                             self._is_train, self._reuse, norm=None,
                             activation=self._activation)
            D = tf.nn.avg_pool(D, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            D = nn.residual2(D, self._num_filters*2, 'd_resnet_1', 3, 1,
                             self._is_train, self._reuse, self._norm,
                             self._activation)
            D = tf.nn.avg_pool(D, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            D = nn.residual2(D, self._num_filters*4, 'd_resnet_3', 3, 1,
                             self._is_train, self._reuse, self._norm,
                             self._activation)
            D = tf.nn.avg_pool(D, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            D = nn.residual2(D, self._num_filters*8, 'd_resnet_4', 3, 1,
                             self._is_train, self._reuse, self._norm,
                             self._activation)
            D = tf.nn.avg_pool(D, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            D = nn.activation_fn(D, self._activation)
            D = tf.nn.avg_pool(D, [1, 8, 8, 1], [1, 8, 8, 1], 'SAME')

            D = nn.linear(tf.reshape(D, [input.get_shape()[0], -1]), 1,
                          name='d_linear_resnet_5')

            self._reuse = True
            self.var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)

            return tf.nn.sigmoid(D), D

    def _convnet(self, input):

        with tf.compat.v1.variable_scope(self.name, reuse=self._reuse):
            D = nn.conv_block(input, self._num_filters, 'd_conv_0', 4, 2,
                              self._is_train, self._reuse, norm=None,
                              activation=self._activation)
            D = nn.conv_block(D, self._num_filters*2, 'd_conv_1', 4, 2,
                              self._is_train, self._reuse, self._norm,
                              self._activation)
            D = nn.conv_block(D, self._num_filters*4, 'd_conv_3', 4, 2,
                              self._is_train, self._reuse, self._norm,
                              self._activation)
            D = nn.conv_block(D, self._num_filters*8, 'd_conv_4', 4, 2,
                              self._is_train, self._reuse, self._norm,
                              self._activation)
            if input.get_shape()[0] == None:
                D = nn.linear(tf.reshape(D,[-1, D.shape[1]*D.shape[2]*D.shape[3]]), 1, name='d_linear_5')
            else:
                D = nn.linear(tf.reshape(D, [input.get_shape()[0], -1]), 1,
                          name='d_linear_5')

            self._reuse = True
            self.var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)

            return tf.nn.sigmoid(D), D
