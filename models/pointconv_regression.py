"""
Modified from PointConv: https://github.com/DylanWusee/pointconv
Author: Ning Hua
Date: May 2021
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from PointConv import feature_encoding_layer, feature_decoding_layer
import time
from functools import reduce
from operator import mul

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 4))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, output_dim, sigma, bn_decay=None, weight_decay = None):

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[..., :3]
    l0_points = point_cloud[..., 3:]
    # print('encode_l0_xyz_point: ', l0_xyz.shape, l0_points.shape) #  (32, 2048, 3) (32, 2048, 3)
    # Feature encoding layers
    # t0 = time.time()
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    # print('en l1: ', time.time()-t0)
    # t0 = time.time()
    # print('encode_l1_xyz_point: ', l1_xyz.shape, l1_points.shape) # (32, 1024, 3) (32, 1024, 64)
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    # print('en l2: ', time.time()-t0)
    # t0 = time.time()
    # print('encode_l2_xyz_point: ', l2_xyz.shape, l2_points.shape) # (32, 256, 3) (32, 256, 128)
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    # print('en l3: ', time.time()-t0)
    # t0 = time.time()
    # print('encode_l3_xyz_point: ', l3_xyz.shape, l3_points.shape) # (32, 64, 3) (32, 64, 256)
    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')
    # print('en l4: ', time.time()-t0)
    # t0 = time.time()
    # print('encode_l4_xyz_point: ', l4_xyz.shape, l4_points.shape) # (32, 36, 3) (32, 36, 512)

    # Feature decoding layers
    l3_points = feature_decoding_layer(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8 * sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    # print('de l1: ', time.time()-t0)
    # t0 = time.time()
    # print('decode_l3_point: ', l3_points.shape) # (32, 64, 512)
    l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    # print('de l2: ', time.time()-t0)
    # t0 = time.time()
    # print('decode_l2_point: ', l2_points.shape) # (32, 256, 256)
    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    # print('de l3: ', time.time()-t0)
    # t0 = time.time()
    # print('decode_l1_point: ', l1_points.shape) # (32, 1024, 128)
    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')
    # print('de l4: ', time.time()-t0)
    # t0 = time.time()
    # print('decode_l0_point: ', l0_points.shape) # (32, 2048, 128)
    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    # print('fc_1: ', net.shape) # (32, 2048, 128)
    end_points['feats'] = net
    # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, output_dim, 1, padding='VALID', activation_fn=tf.nn.sigmoid, weight_decay=weight_decay, scope='fc2')
    # print('fc: ', time.time()-t0)
    # print('fc_2: ', net.shape) # (32, 2048, 10)
    # print('end_point:', end_points['feats'].shape) # (32, 2048, 128)

    return tf.squeeze(net, axis=-1), end_points

def get_model_3layers(point_cloud, is_training, output_dim, sigma, bn_decay=None, weight_decay = None):

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[..., :3]
    l0_points = point_cloud[..., 3:]
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')

    l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')
    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.conv1d(net, output_dim, 1, padding='VALID', activation_fn=tf.nn.sigmoid, weight_decay=weight_decay, scope='fc2')

    return tf.squeeze(net, axis=-1), end_points

def get_model_2layers(point_cloud, is_training, output_dim, sigma, bn_decay=None, weight_decay = None):

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[..., :3]
    l0_points = point_cloud[..., 3:]
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    
    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.conv1d(net, output_dim, 1, padding='VALID', activation_fn=tf.nn.sigmoid, weight_decay=weight_decay, scope='fc2')

    return tf.squeeze(net, axis=-1), end_points

def get_model_1layers(point_cloud, is_training, output_dim, sigma, bn_decay=None, weight_decay = None):

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[..., :3]
    l0_points = point_cloud[..., 3:]
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')

    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.conv1d(net, output_dim, 1, padding='VALID', activation_fn=tf.nn.sigmoid, weight_decay=weight_decay, scope='fc2')

    return tf.squeeze(net, axis=-1), end_points


def get_l2_rel_loss(pred, y_true):
    batch_size = pred.shape[0]
    diff_norms = tf.norm(tf.reshape(pred, (batch_size, -1)) - tf.reshape(y_true, (batch_size, -1)), ord=2, axis=1)
    y_norms = tf.norm(tf.reshape(y_true, (batch_size, -1)), ord=2, axis=1)
    regression_l2_rel_loss = tf.reduce_sum(diff_norms/y_norms)
    return regression_l2_rel_loss


if __name__=='__main__':

    with tf.Graph().as_default():

        inputs = tf.zeros((1,2048,7))
        t0 = time.time()
        net, _ = get_model(inputs, tf.constant(True), 1, 1.0)
        print('time: ', time.time()-t0)
        print(net.shape)
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print(num_params)
