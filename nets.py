import os
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import tensorflow.contrib.slim as slim
from keras.layers import Input, LocallyConnected1D, Reshape, Permute, Conv2D, Add
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from functools import partial
import dnnlib.tflib as tflib


# Inception-Resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn: net = activation_fn(net)
    return net


# Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7], scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1], scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn: net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3], scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1], scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn: net = activation_fn(net)
    return net


def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3, scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net


def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3, scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
    return net


def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train, dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None, 
                        scope='InceptionResnetV1'):
    """
    Creates the Inception Resnet V1 model.
    Args:
        inputs: a 4-D tensor of size [batch_size, height, width, 3].
        num_classes: number of predicted classes.
        is_training: whether is training or not.
        dropout_keep_prob: float, the fraction to keep before final layer.
        reuse: whether or not the network and its variables should be reused. To be able to reuse 'scope' must be given.
        scope: Optional variable_scope.
    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.
    """
    end_points = {}
  
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID', scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID', scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net
                
                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net
        
                # Reduction-A
                with tf.variable_scope('Mixed_6a'): net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net
                
                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net
                
                # Reduction-B
                with tf.variable_scope('Mixed_7a'): net = reduction_b(net)
                end_points['Mixed_7a'] = net
                
                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net
                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net
                
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')
                    end_points['PreLogitsFlatten'] = net
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
    return net, end_points


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size, tiled_dlatent, model_scale=18):
    if tiled_dlatent:
        low_dim_dlatent = tf.get_variable('learnable_dlatents',
            shape=(batch_size, 512),
            dtype='float32',
            initializer=tf.initializers.random_normal())
        return tf.tile(tf.expand_dims(low_dim_dlatent, axis=1), [1, model_scale, 1])
    else:
        return tf.get_variable('learnable_dlatents',
            shape=(batch_size, model_scale, 512),
            dtype='float32',
            initializer=tf.initializers.random_normal())


class Generator:
    def __init__(self, model, batch_size=1, clipping_threshold=2, tiled_dlatent=True, model_res=1024, randomize_noise=False):
        self.batch_size = batch_size
        self.tiled_dlatent = tiled_dlatent
        self.model_scale = int(2*(math.log(model_res,2)-1))

        if tiled_dlatent:
            self.initial_dlatents = np.zeros((self.batch_size, 512))
            model.components.synthesis.run(np.zeros((self.batch_size, self.model_scale, 512)),
                randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, tiled_dlatent=True), partial(create_stub, batch_size=batch_size)],
                structure='fixed')
        else:
            self.initial_dlatents = np.zeros((self.batch_size, self.model_scale, 512))
            model.components.synthesis.run(self.initial_dlatents,
                randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, tiled_dlatent=False, model_scale=self.model_scale), partial(create_stub, batch_size=batch_size)],
                structure='fixed')

        self.dlatent_avg_def = model.get_var('dlatent_avg')
        self.reset_dlatent_avg()
        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()
        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
        self.set_dlatents(self.initial_dlatents)

        try:
            self.generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0')
        except KeyError:
            # If we loaded only Gs and didn't load G or D, then scope "G_synthesis_1" won't exist in the graph.
            self.generator_output = self.graph.get_tensor_by_name('G_synthesis/_Run/concat:0')
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

        # Implement stochastic clipping similar to what is described in https://arxiv.org/abs/1702.04782
        # (Slightly different in that the latent space is normal gaussian here and was uniform in [-1, 1] in that paper,
        # so we clip any vector components outside of [-2, 2]. It seems fine, but I haven't done an ablation check.)
        clipping_mask = tf.math.logical_or(self.dlatent_variable > clipping_threshold, self.dlatent_variable < -clipping_threshold)
        clipped_values = tf.where(clipping_mask, tf.random_normal(shape=self.dlatent_variable.shape), self.dlatent_variable)
        self.stochastic_clip_op = tf.assign(self.dlatent_variable, clipped_values)

    def reset_dlatents(self):
        self.set_dlatents(self.initial_dlatents)

    def set_dlatents(self, dlatents):
        if self.tiled_dlatent:
            if (dlatents.shape != (self.batch_size, 512)) and (dlatents.shape[1] != 512):
                dlatents = np.mean(dlatents, axis=1)
            if (dlatents.shape != (self.batch_size, 512)):
                dlatents = np.vstack([dlatents, np.zeros((self.batch_size-dlatents.shape[0], 512))])
            assert (dlatents.shape == (self.batch_size, 512))
        else:
            if (dlatents.shape[1] > self.model_scale):
                dlatents = dlatents[:,:self.model_scale,:]
            if (dlatents.shape != (self.batch_size, self.model_scale, 512)):
                dlatents = np.vstack([dlatents, np.zeros((self.batch_size-dlatents.shape[0], self.model_scale, 512))])
            assert (dlatents.shape == (self.batch_size, self.model_scale, 512))
        self.sess.run(tf.assign(self.dlatent_variable, dlatents))

    def stochastic_clip_dlatents(self):
        self.sess.run(self.stochastic_clip_op)

    def get_dlatents(self):
        return self.sess.run(self.dlatent_variable)

    def get_dlatent_avg(self):
        return self.dlatent_avg

    def set_dlatent_avg(self, dlatent_avg):
        self.dlatent_avg = dlatent_avg

    def reset_dlatent_avg(self):
        self.dlatent_avg = self.dlatent_avg_def

    def generate_images(self, dlatents=None):
        if dlatents is not None:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)

    def get_output(self, dlatent_tensor):
        tf.assign(self.dlatent_variable, dlatent_tensor)
        generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0')
        generated_image = tflib.convert_images_to_uint8(generator_output, nchw_to_nhwc=True, uint8_cast=False)
        return tf.saturate_cast(generated_image, tf.uint8)