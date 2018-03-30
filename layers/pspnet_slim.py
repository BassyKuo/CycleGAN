import functools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def pspnet50(inputs, num_classes, #for segmentation
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02), 
                    biases_initializer=tf.zeros_initializer, 
                    phase_train=False, reuse=False, 
                    scope='pspnet50', segment_scope='segment'):
    """
    Return:
      - conv5_3 (the last layer before segmentation computation)
      - segment (name: <scope>/<segment_scope>/*)
      - endpoints
    """
    outputs_collections = scope+'_endpoints'
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.avg_pool2d, slim.max_pool2d, slim.flatten, slim.fully_connected], 
                outputs_collections=outputs_collections):
            with slim.arg_scope([slim.conv2d], padding='VALID',
                            activation_fn=None,
                            weights_initializer=weights_initializer,
                            biases_initializer=None,
                            trainable=phase_train):
                with slim.arg_scope([slim.batch_norm],
                            decay=0.9, 
                            epsilon=1e-4, 
                            activation_fn=slim.nn.relu, 
                            is_training=phase_train, 
                            trainable=phase_train, 
                            fused=True):
                    # --- resnet50 ---
                    net = slim.conv2d(net, 64, 3, 2, padding='SAME', scope='conv1_1_3x3_s2')
                    net = slim.batch_norm(net, scope='conv1_1_3x3_s2_bn')
                    net = slim.conv2d(net, 64, 3, 1, padding='SAME', scope='conv1_2_3x3')
                    net = slim.batch_norm(net, scope='conv1_2_3x3_bn')
                    net = slim.conv2d(net, 128, 3, 1, padding='SAME', scope='conv1_3_3x3')
                    net = slim.batch_norm(net, scope='conv1_3_3x3_bn')
                    net = slim.max_pool2d(net, 3, 2, padding='SAME', scope='pool1_3x3_s2')

                    identity = slim.conv2d(net, 256, 1, 1, scope='conv2_1_1x1_proj')
                    identity = slim.batch_norm(identity, activation_fn=None, scope='conv2_1_1x1_proj_bn')



                    for idx in range(0,3):
                        if idx != 0:
                            with tf.variable_scope('conv2_{}'.format(idx)):
                                net = tf.add_n([identity, net], name='Add')
                                net = slim.nn.relu(net, name='Relu'); identity = net
                        with tf.variable_scope('conv2_{}'.format(idx+1)):
                            net = slim.conv2d(net, 64, 1, 1, scope='1x1_reduce')
                            net = slim.batch_norm(net, scope='1x1_reduce_bn')
                            net = zero_padding(net, paddings=1, name='padding')
                            net = slim.conv2d(net, 64, 3, 1, scope='3x3')
                            net = slim.batch_norm(net, scope='3x3_bn')
                            net = slim.conv2d(net, 256, 1, 1, scope='1x1_increase')
                            net = slim.batch_norm(net, activation_fn=None, scope='1x1_increase_bn')


                    net = tf.add_n([identity, net], name='conv2_3/Add')
                    net = slim.nn.relu(net, name='conv2_3/Relu')
                    identity = slim.conv2d(net, 512, 1, 2, scope='conv3_1_1x1_proj')
                    identity = slim.batch_norm(identity, activation_fn=None, scope='conv3_1_1x1_proj_bn')

                    for idx in range(0,4):
                        if idx != 0:
                            with tf.variable_scope('conv3_{}'.format(idx)):
                                net = tf.add_n([identity, net], name='Add')
                                net = slim.nn.relu(net, name='Relu'); identity = net
                        with tf.variable_scope('conv3_{}'.format(idx+1)):
                            s = 2 if idx == 0 else 1
                            net = slim.conv2d(net, 128, 1, s, scope='1x1_reduce')
                            net = slim.batch_norm(net, scope='1x1_reduce_bn')
                            net = zero_padding(net, paddings=1, name='padding')
                            net = slim.conv2d(net, 128, 3, 1, scope='3x3')
                            net = slim.batch_norm(net, scope='3x3_bn')
                            net = slim.conv2d(net, 512, 1, 1, scope='1x1_increase')
                            net = slim.batch_norm(net, activation_fn=None, scope='1x1_increase_bn')


                    net = tf.add_n([identity, net], name='conv3_4/Add')
                    net = slim.nn.relu(net, name='conv3_4/Relu')
                    identity = slim.conv2d(net, 1024, 1, 1, scope='conv4_1_1x1_proj')
                    identity = slim.batch_norm(identity, scope='conv4_1_1x1_proj_bn')

                    for idx in range(0,6):
                        if idx != 0:
                            with tf.variable_scope('conv4_{}'.format(idx)):
                                net = tf.add_n([identity, net], name='Add')
                                net = slim.nn.relu(net, name='Relu'); identity = net
                        with tf.variable_scope('conv4_{}'.format(idx+1)):
                            net = slim.conv2d(net, 256, 1, 1, scope='1x1_reduce')
                            net = slim.batch_norm(net, scope='1x1_reduce_bn')
                            net = zero_padding(net, paddings=2, name='padding')
                            net = atrous_conv2d(net, 256, 3, 2, scope='3x3_atrous')
                            net = slim.batch_norm(net, scope='3x3_atrous_bn')
                            net = slim.conv2d(net, 1024, 1, 1, scope='1x1_increase')
                            net = slim.batch_norm(net, activation_fn=None, scope='1x1_increase_bn')


                    net = tf.add_n([identity, net], name='conv4_6/Add')
                    net = slim.nn.relu(net, name='conv4_6/Relu')
                    identity = slim.conv2d(net, 2048, 1, 1, scope='conv5_1_1x1_proj')
                    identity = slim.batch_norm(identity, activation_fn=None, scope='conv5_1_1x1_proj_bn')

                    for idx in range(0,3):
                        if idx != 0:
                            with tf.variable_scope('conv5_{}'.format(idx)):
                                net = tf.add_n([identity, net], name='Add')
                                net = slim.nn.relu(net, name='Relu'); identity = net
                        with tf.variable_scope('conv5_{}'.format(idx+1)):
                            net = slim.conv2d(net, 512, 1, 1, scope='1x1_reduce')
                            net = slim.batch_norm(net, scope='1x1_reduce_bn')
                            net = zero_padding(net, paddings=4, name='padding')
                            net = atrous_conv2d(net, 512, 3, 4, scope='3x3_atrous')
                            net = slim.batch_norm(net, scope='3x3_atrous_bn')
                            net = slim.conv2d(net, 2048, 1, 1, scope='1x1_increase')
                            net = slim.batch_norm(net, activation_fn=None, scope='1x1_increase_bn')


                    net = tf.add_n([identity, net], name='conv5_3/Add')
                    conv5_3 = slim.nn.relu(net, name='conv5_3/Relu')
                    resize = tf.shape(conv5_3)[1:3]

                    # --- segmentation ---
                    nets = [conv5_3]
                    with tf.variable_scope(segment_scope):
                        for dim in [6,3,2,1]:
                            with tf.variable_scope('conv5_3_pool{}'.format(dim)):
                                net = slim.avg_pool2d(conv5_3, [int(dim)]*2, int(dim), padding='VALID')
                                net = slim.conv2d(net, 512, 1, 1, scope='1x1_reduce')
                                net = slim.batch_norm(net, scope='1x1_reduce_bn')
                                net = tf.image.resize_bilinear(net, resize, align_corners=True, name='Interp')
                                nets.append(net)
                        segment = tf.concat(nets, axis=-1, name='conv5_3_concat')
                        segment = slim.conv2d(segment, 512, 3, 1, padding='SAME', scope='conv5_4')
                        segment = slim.batch_norm(segment, scope='conv5_4_bn')
                        segment = slim.conv2d(segment, num_classes, 1, 1, biases_initializer=biases_initializer, scope='conv6')

                    ## --- ResNet-50 ---
                    #with tf.variable_scope('resnet50', reuse=tf.AUTO_REUSE):
                        #resnet50 = slim.avg_pool2d(conv5_3, [7,7], 7, padding='VALID', scope='avg_pool')
                        #resnet50 = slim.flatten(resnet50, scope='flatten')
                        #resnet50 = slim.fully_connected(resnet50, num_classes, activation_fn=None, scope='fc2')

    return conv5_3, segment, tf.get_collection(outputs_collections)

def zero_padding(inputs, paddings, name='zero_padding'):
    pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
    return tf.pad(inputs, paddings=pad_mat, name=name)

def atrous_conv2d(inputs, num_outputs, kernel_size, dilation=2, **kwargs):
    kwargs.update({'stride':1, 'rate':dilation})
    return slim.conv2d(inputs, num_outputs, kernel_size, **kwargs)
