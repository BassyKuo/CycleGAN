#!/usr/bin/env python3
import math
import numpy as np
import tensorflow as tf                 # tf.__version__ : 1.4
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.framework import tensor_shape
#from tensorflow.contrib.keras.python.keras import backend as K

VarianceScaling = tf.contrib.keras.initializers.VarianceScaling

class WeightScaleInitializer(Initializer):
    """Initializer that the weights normalized by he_normal standard deviation. (Tero Karras et al., 2017)

    w_new = w / \sqrt( \mean( w^2 ) )  where w is sampling from he_normal(stddev=\sqrt(2. / fan_in)).

    ** scale = \sqrt( \mean( w^2 ) ) = \sqrt(2. / fan_in) **

     Arguments:
        seed: A Python integer. Used to seed the random generator.
    Returns:
        An initializer.
    References:
        he_normal: He et al., http://arxiv.org/abs/1502.01852
        ws_normal: Tero Karras et al., https://arxiv.org/abs/1710.10196
    """
    def __init__(self, scale=1.0, mode='fan_in', distribution='normal', seed=None):
        if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument: '
                             'expected on of {"fan_in", "fan_out", "fan_avg"} '
                             'but got', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            stddev = math.sqrt(scale)
            return tf.truncated_normal(shape, 0., stddev, dtype=dtype, seed=self.seed) / stddev
        else:
            limit = math.sqrt(3. * scale)
            return tf.random_uniform(shape, -limit, limit, dtype=dtype, seed=self.seed) / limit

    def get_config(self):
        return {
                'scale': self.scale,
                'mode': self.mode,
                'distribution': self.distribution,
                'seed': self.seed
                }

def ws_normal(seed=None):
    """Initializer that the weights normalized by he_normal standard deviation. (Tero Karras et al., 2017)

        w_new = w / \sqrt( \mean( w^2 ) )

        where w is sampling from he_normal(stddev=\sqrt(2. / fan_in)).

    ** scale = \sqrt( \mean( w^2 ) ) = \sqrt(2. / fan_in) **

    Arguments:
        seed: A Python integer. Used to seed the random generator.
    Returns:
        An initializer.
    References:
        he_normal: He et al., http://arxiv.org/abs/1502.01852
        ws_normal: Tero Karras et al., https://arxiv.org/abs/1710.10196
    """
    return WeightScaleInitializer(scale=2., mode='fan_in', distribution='normal', seed=seed)

def _compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.
    Arguments:
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).
    Returns:
        A tuple of scalars, `(fan_in, fan_out)`.
    Raises:
        ValueError: in case of invalid `data_format` argument.
    """
    shape = tensor_shape.TensorShape(shape).as_list()
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = math.sqrt(np.prod(shape))
        fan_out = math.sqrt(np.prod(shape))
    return fan_in, fan_out
