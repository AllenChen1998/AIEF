import os
import pickle
import time
import PIL.Image
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib import networks_stylegan as style
from nets import inference

paths = {
    'stylegan_celeba':     'models/stylegan/karras2019stylegan-celebahq-1024x1024.pkl',
    'facenet':             'models/facenet/model-20180402-114759.ckpt-275'
}

def build(batch_size, saturate_threshold):
    tf.InteractiveSession().as_default()
    sess = tf.get_default_session()

    weights = tf.get_variable(name='weight_w', shape=(batch_size, 18, 512), dtype=tf.float32, initializer=tf.ones_initializer())
    z = tf.get_variable(name='init_w', shape=(batch_size, 18, 512), dtype=tf.float32, initializer=tf.zeros_initializer())
    delta_z = tf.get_variable(name='delta_w', shape=(batch_size, 18, 512), dtype=tf.float32, initializer=tf.zeros_initializer())
    weights_sum = tf.reduce_sum(tf.reshape(weights, [batch_size, -1]), axis=1)
    adv_z = z + delta_z * weights / tf.tile(tf.reshape(weights_sum, (-1, 1, 1)), (1, 18, 512)) * 18 * 512

    image, loss = {}, {}
    image['real'] = tf.placeholder(tf.float32, shape=[adv_z.shape[0], 1024, 1024, 3])

    noise = []
    for layer_idx in range(18):
        res = layer_idx // 2 + 2
        shape = [batch_size, 1, 2**res, 2**res]
        noise.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=True))

    print('\nLoading Decoder...')
    _, _, Gs = pickle.load(open(paths['stylegan_celeba'], "rb"))
    with tf.name_scope('decoder'):
        image['adv_tmp'] = Gs.components.synthesis.get_output_for(adv_z, noise_inputs=noise, function=style.G_synthesis)
        image['adv'] = tflib.convert_images_to_01(image['adv_tmp'], nchw_to_nhwc=True)
        image['adv255'] = tflib.convert_images_to_uint8(image['adv_tmp'], nchw_to_nhwc=True)

    print('Loading Facenet...\n')
    with tf.name_scope('facenet'):
        facenet = Facenet(512)
        loss['face'], facenet_variable, _ = facenet.build(image['real'], image['adv'])
        tf.train.Saver(var_list=facenet_variable).restore(sess, paths['facenet'])

    loss['image'] = tf.abs(image['adv'] * 256 - image['real'])
    loss['face_sat'] = tf.nn.relu(saturate_threshold - loss['face'])

    optimizer = {}
    optimizer['encoder'] = tf.train.AdamOptimizer(0.01).minimize(loss['image'], var_list=[z] + noise)
    optimizer['delta'] = tf.train.AdamOptimizer(0.005).minimize(loss['face_sat'], var_list=[delta_z])
    optimizer['weight'] = tf.train.AdamOptimizer(0.2).minimize(loss['image'], var_list=[weights])
    initialize_uninitialized(sess)
    var_list = [z] + noise + [delta_z] + [weights]
    return sess, image, loss, optimizer, var_list

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars): sess.run(tf.variables_initializer(not_initialized_vars))

def output(value_dict, stream=None, bit=3):
    output_str = ''
    for key, value in value_dict.items():
        if isinstance(value, list): value = value[-1]
        if isinstance(value, float) or isinstance(value, np.float32) or isinstance(value, np.float64): value = round(value, bit)
        output_str += '[ ' + str(key) + ' ' + str(value) + ' ] '
    print(output_str, end='\r')
    if stream is not None: print(output_str, file=stream)

def get_time(deviation=0): return time.strftime(
    '%Y-%m-%d %H-%M-%S', time.localtime(time.time()-deviation))

class Facenet:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

    def build(self, x1, x2):
        x1, x2 = tf.image.resize_bilinear(x1, [160, 160]), tf.image.resize_bilinear(x2, [160, 160])
        batch_num = int(x1.shape[0])
        loss_face = []

        for batch_id in range(batch_num):
            prewhited1, prewhited2 = self.prewhiten(tf.expand_dims(x1[batch_id], 0)), self.prewhiten(tf.expand_dims(x2[batch_id], 0))
            prelogits, end_points = inference(tf.concat((prewhited1, prewhited2), axis=0), keep_probability=1, phase_train=False, bottleneck_layer_size=self.embedding_size, reuse=tf.AUTO_REUSE)
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            loss_face.append(tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1])) + 1e-6))
        return tf.convert_to_tensor(loss_face), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1'), end_points

    def prewhiten(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1, 2, 3])
        std = tf.sqrt(var)
        size = tf.cast(x.shape[1]*x.shape[2]*x.shape[3], tf.float32)
        std_adj = tf.maximum(std, 1.0/tf.sqrt(size))
        y = tf.multiply(tf.subtract(x, mean), 1/std_adj)
        return y
