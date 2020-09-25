# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from tensorpack.tfutils.argscope import argscope
from tensorpack.models import (
    Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm, FullyConnected)
from tensorpack.utils.develop import log_deprecated
from tensorpack.models.common import layer_register
import numpy as np
import functools


# activation functions
def gelu_fn(x, name):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  with tf.compat.v1.name_scope(name):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def smoothrelu_fn(x, name):
  """SmoothReLU.

  This is another smoother version of the RELU.
  Original paper: https://arxiv.org/pdf/2006.14536.pdf

  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the SmoothReLU activation applied.
  """
  with tf.compat.v1.name_scope(name):
    beta_raw = tf.get_variable('smoothrelu_beta', [1],
                          initializer=tf.constant_initializer(20.0),
                          dtype=tf.float32)
    beta = tf.math.square(beta_raw)
    # safe_log implementation follows https://github.com/tensorflow/tensorflow/issues/38349
    # safe_log = tf.math.log(tf.where(x > 0., beta * x + 1., tf.ones_like(x)))
    # return tf.where((x > 0.), x - (1./beta) * safe_log, tf.zeros_like(x))
    # a faster implementation
    return tf.nn.relu(x - (1./beta) * tf.math.log(tf.abs(x) * beta + 1.))

def mish_fn(x, name):
  """Self Regularized Non-Monotonic Activation Function.

  Original paper: https://arxiv.org/pdf/1908.08681.pdf

  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the Mish activation applied.
  """
  with tf.compat.v1.name_scope(name):
    return x * tf.tanh(tf.math.softplus(x))
     

activation_list = {'relu': tf.nn.relu,
                   'silu': tf.nn.swish,
                   'softplus': tf.nn.softplus,
                   'elu': tf.nn.elu,
                   'gelu': gelu_fn,
                   'smoothrelu': smoothrelu_fn,
                   'mish': mish_fn,
}


@layer_register(use_scope=None)
def BNActivation(x, activation_name='relu', name=None):
    """
    A shorthand of BatchNormalization + ReLU.

    Args:
        x (tf.Tensor): the input
        name: deprecated, don't use.
    """
    if name is not None:
        log_deprecated("BNActivation(name=...)", "The output tensor will be named `output`.")

    x = BatchNorm('bn', x)

    activation_fn = activation_list[activation_name]
    x = activation_fn(x, name=name)
    return x


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def fixed_padding(inputs, kernel_size, data_format='channels_first'):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def resnet_bottleneck(l, ch_out, stride, group=1, res2_bottleneck=64, activation_name='relu'):
    """
    Args:
        group (int): the number of groups for resnext
        res2_bottleneck (int): the number of channels in res2 bottleneck.
    The default corresponds to ResNeXt 1x64d, i.e. vanilla ResNet.
    """
    ch_factor = res2_bottleneck * group // 64
    shortcut = l
    l = Conv2D('conv1', l, ch_out * ch_factor, 1, strides=1, activation=functools.partial(BNActivation, activation_name=activation_name))
    # this padding manner follows https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L358
    if stride == 2:
        l = fixed_padding(l, 3)
    l = Conv2D('conv2', l, ch_out * ch_factor, 3, strides=stride, activation=functools.partial(BNActivation, activation_name=activation_name), split=group, padding=('same' if stride == 1 else 'valid'))
    """
    ImageNet in 1 Hour, Sec 5.1:
    the stride-2 convolutions are on 3×3 layers instead of on 1×1 layers
    """
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    """
    ImageNet in 1 Hour, Sec 5.1: each residual block's last BN where γ is initialized to be 0
    """
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    activation_fn = activation_list[activation_name]
    return activation_fn(ret, name='block_output')
    # return tf.nn.swish(ret, name='block_output')


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                current_stride = stride if i == 0 else 1
                l = block_func(l, features, current_stride)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func, activation_name='relu'):
    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, use_bias=False,
                     kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # this padding manner follows https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L358
        l = fixed_padding(image, 7)
        l = Conv2D('conv0', l, 64, 7, strides=2, activation=functools.partial(BNActivation, activation_name=activation_name), padding='valid')
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        """
        ImageNet in 1 Hour, Sec 5.1:
        The 1000-way fully-connected layer is initialized by
        drawing weights from a zero-mean Gaussian with standard deviation of 0.01
        """
    return logits
