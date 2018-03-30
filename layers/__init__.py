import functools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def ConvAvgPool(inputs, num_outputs, kernel_size, stride=1, padding='SAME', rate=1, name='ConvMeanPool', **kwargs):
    with tf.variable_scope(name):
        output = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding, rate=rate, **kwargs)
        output = slim.avg_pool2d(output, kernel_size=2, stride=2, padding='VALID')
    return output

def AvgPoolConv(inputs, num_outputs, kernel_size, stride=1, padding='SAME', rate=1, name='AvgPoolConv', **kwargs):
    with tf.variable_scope(name):
        output = slim.avg_pool2d(inputs, kernel_size=2, stride=2, padding='VALID')
        output = slim.conv2d(output, num_outputs, kernel_size, stride=stride, padding=padding, rate=rate, **kwargs)
    return output

def UpsampleConv(inputs, num_outputs, kernel_size, stride=1, padding='SAME', rate=1, name='UpsampleConv', **kwargs):
    with tf.variable_scope(name):
        h,w = inputs.shape.as_list()[1:3]
        output = tf.image.resize_nearest_neighbor(inputs, size=[2*h, 2*w])
        output = slim.conv2d(output, num_outputs, kernel_size, stride=stride, padding=padding, rate=rate, **kwargs)
    return output


def ResidualBlock(inputs, num_outputs, kernel_size, stride=1, padding="SAME", activation_fn=slim.nn.relu, normalization=slim.batch_norm, 
        weights_initializer=tf.truncated_normal_initializer(stddev=0.02), biases_initializer=tf.zeros_initializer, 
        type=None, scope='ResidualBlock'):
    """
    Args:
        - type: sample type. [None, "up", "down"]
    """
    weight_init = weights_initializer
    biases_init = biases_initializer
    with tf.variable_scope(scope):
        input_dim = inputs.shape.as_list()[-1]
        if type == None:
            if input_dim == num_outputs:
                conv_shortcut = tf.identity
            else:
                conv_shortcut = functools.partial(slim.conv2d, num_outputs=num_outputs, kernel_size=1, weights_initializer=weight_init, biases_initializer=biases_init)
            conv_1          = functools.partial(slim.conv2d, num_outputs=input_dim)
            conv_2          = functools.partial(slim.conv2d, num_outputs=num_outputs)
        elif type == 'up':
            conv_shortcut   = functools.partial(UpsampleConv, num_outputs=num_outputs, kernel_size=1, weights_initializer=weight_init, biases_initializer=biases_init)
            conv_1          = functools.partial(UpsampleConv, num_outputs=num_outputs)
            conv_2          = functools.partial(slim.conv2d,  num_outputs=num_outputs)
        elif type == 'down':
            conv_shortcut   = functools.partial(AvgPoolConv, num_outputs=num_outputs, kernel_size=1, weights_initializer=weight_init, biases_initializer=biases_init)
            conv_1          = functools.partial(slim.conv2d, num_outputs=input_dim)
            conv_2          = functools.partial(ConvAvgPool, num_outputs=num_outputs)
        else:
            raise Exception('invalid sample type.')

        shortcut = conv_shortcut(inputs, name='Shortcut')

        output = inputs
        output = normalization(output)
        output = activation_fn(output)
        output = conv_1(output, kernel_size=kernel_size, stride=stride, weights_initializer=weight_init, biases_initializer=None)
        otuput = normalization(output)
        otuput = activation_fn(output)
        output = conv_2(output, kernel_size=kernel_size, stride=stride, weights_initializer=weight_init)

        return shortcut + output

def SPPLayers(inputs, pooling_dims=[6,3,2,1], scope='SPPLayers'):
    resize = tf.shape(inputs)[1:3]
    nets = [inputs]
    for dim in pooling_dims:
        with tf.variable_scope('{}_pool{}'.format(scope, dim)):
            net = slim.avg_pool2d(inputs, [int(dim)]*2, int(dim), padding='VALID')
            net = slim.conv2d(net, 512, 1, 1, scope='1x1_reduce')
            net = slim.batch_norm(net, scope='1x1_reduce_bn')
            net = tf.image.resize_bilinear(net, resize, align_corners=True, name='Interp')
            nets.append(net)
    net = tf.concat(nets, axis=-1, name='{}_concat'.format(scope))
    return net

def Mask(segment, size, index=1, name='Mask'):
    assert len(size) == 2
    with tf.variable_scope(name):
        mask = tf.image.resize_bilinear(segment, size)
        mask = tf.argmax(mask, axis=-1) # generate segment indices matrix
        mask = tf.where(tf.not_equal(mask, index), tf.zeros_like(mask), tf.ones_like(mask))
    return mask
