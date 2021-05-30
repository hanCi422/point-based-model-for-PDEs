"""
Modified from PointConv: https://github.com/DylanWusee/pointconv
Author: Ning Hua
Date: May 2021
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import importlib
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'PDE_datasets'))
import poisson_2d
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointconv_regression', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--ntrain', default=1000, help='number of train [default: 1000]')
parser.add_argument('--ntest', default=1000, help='number of test [default: 1000]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
EPOCH_CNT_WHOLE = 0

BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
NTRAIN = FLAGS.ntrain
NTEST = FLAGS.ntest
BANDWIDTH = 0.05

MODEL = importlib.import_module(FLAGS.model) # import network module

OUTPUT_DIM = 1

print("start loading train data ...")
DATA_PATH = 'XXXXXX/poisson_2d_K3_train.npy'
TRAIN_DATASET = poisson_2d.Poisson_2d(root=DATA_PATH, nsample=NTRAIN)
print("start loading test data ...")
DATA_PATH = 'XXXXXX/poisson_2d_K3_test2.npy'
TEST_DATASET = poisson_2d.Poisson_2d_test(root=DATA_PATH, nsample=NTEST)
NUM_POINT = TEST_DATASET.pointnumber()

def test():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, OUTPUT_DIM, BANDWIDTH)
        loss = MODEL.get_l2_rel_loss(TRAIN_DATASET.denormalize_y(pred), TRAIN_DATASET.denormalize_y(labels_pl))
        saver = tf.train.Saver()
    
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    MODEL_PATH = '.ckpt'
    saver.restore(sess, MODEL_PATH)
    print("Model restored.")
    
    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}
    print('BATCH SIZE: ', BATCH_SIZE)
    loss_valid, record = eval_and_record(sess, ops)
    print('TEST L2 REL Loss: %.8f'%(loss_valid))
    # np.save('poisson_pointconv_K3_record.npy', record)


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_x = np.zeros((bsize, NUM_POINT, 4), dtype=np.float32)
    batch_y = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        x, y = dataset[idxs[i+start_idx]]
        batch_x[i,...] = x
        batch_y[i,:] = y
    return batch_x, batch_y

def eval_and_record(sess, ops):
    """ ops: dict mapping from string to tf ops """
    record = np.zeros((NTEST, NUM_POINT, 3), dtype=np.float32)
    is_training = False
    valid_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = int(len(TEST_DATASET)/BATCH_SIZE)

    total_regression_l2_rel_loss = 0
    
    print('---- TEST ----')
   
    for batch_idx in range(num_batches):
        # warm-up
        if batch_idx == 1:
            t0 = time.time()
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_x, batch_y = get_batch(TEST_DATASET, valid_idxs, start_idx, end_idx)
        batch_x_copy = batch_x.copy()
        batch_x, batch_y = TRAIN_DATASET.normalize(batch_x, batch_y)

        feed_dict = {ops['pointclouds_pl']: batch_x,
                    ops['labels_pl']: batch_y,
                    ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        batch_y = TRAIN_DATASET.denormalize_y(batch_y)
        pred_val = TRAIN_DATASET.denormalize_y(pred_val)

        total_regression_l2_rel_loss += loss_val
        record[start_idx:end_idx, :, :] = np.concatenate((batch_x_copy[:,:,3][...,None], batch_y[...,None], pred_val[...,None]), -1)
        print(loss_val)
    print('time of eval for one batch:', (time.time()-t0)/(NTEST-1))
    return total_regression_l2_rel_loss / NTEST, record

if __name__ == "__main__":
    test()

