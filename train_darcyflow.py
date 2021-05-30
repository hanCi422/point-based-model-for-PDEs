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
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'PDE_datasets'))
import darcyflow_2d
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointconv_regression', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--ntrain', default=1024, help='number of train [default: 1024]')
parser.add_argument('--ntest', default=100, help='number of test [default: 100]')
parser.add_argument('--max_point', type=int, default=7225, help='allowed input point number [default: 7225]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--downsample_rate', type=int, default=5, help='downsample for dataset')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
MAX_POINT = FLAGS.max_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NTRAIN = FLAGS.ntrain
NTEST = FLAGS.ntest
BANDWIDTH = 0.05
R = FLAGS.downsample_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
Point_Util = os.path.join(BASE_DIR, 'utils', 'pointconv_util.py')
LOG_DIR = FLAGS.log_dir + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


OUTPUT_DIM = 1

DATA_PATH = 'XXXXXX/piececonst_r421_N1024_smooth1.mat'
print("start loading training data ...")
TRAIN_DATASET = darcyflow_2d.Darcyflow_2d(root=DATA_PATH, r=R, nsample=NTRAIN)
print("start loading validation data ...")
DATA_PATH = 'XXXXXX/piececonst_r421_N1024_smooth2.mat'
VALID_DATASET = darcyflow_2d.Darcyflow_2d(root=DATA_PATH, r=R, nsample=NTEST)
NUM_POINT = TRAIN_DATASET.pointnumber()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, MAX_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            print("--- Get model and loss")
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, OUTPUT_DIM, BANDWIDTH, bn_decay=bn_decay)
            loss = MODEL.get_l2_rel_loss(TRAIN_DATASET.denormalize_y(pred), TRAIN_DATASET.denormalize_y(labels_pl))

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)

            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'step': batch,
               'end_points': end_points}

        best_valid = 1e5
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            start_time = time.time()
            loss_train = train_one_epoch(sess, ops)
            end_time = time.time()
            log_string('one epoch time: %.4f'%(end_time - start_time))
            log_string('TRAIN L2 REL Loss: %.8f'%(loss_train))
            loss_valid = eval_one_epoch(sess, ops)
            log_string('VALID L2 REL Loss: %.8f'%(loss_valid))
            if loss_valid < best_valid:
                best_valid = loss_valid
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_darcyflow2d.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_x = np.zeros((bsize, NUM_POINT, 4), dtype=np.float32)
    batch_y = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        x, y = dataset[idxs[i+start_idx]]
        batch_x[i,...] = x
        batch_y[i,:] = y
    return batch_x, batch_y

def train_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = int(len(TRAIN_DATASET)/BATCH_SIZE)
    
    log_string(str(datetime.now()))

    total_regression_l2_rel_loss = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_x, batch_y = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        batch_x, batch_y = TRAIN_DATASET.normalize(batch_x, batch_y)

        feed_dict = {ops['pointclouds_pl']: batch_x,
                    ops['labels_pl']: batch_y,
                    ops['is_training_pl']: is_training,}
        step, _, loss_val, pred_val = sess.run([ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        total_regression_l2_rel_loss += loss_val

    return total_regression_l2_rel_loss / NTRAIN


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    valid_idxs = np.arange(0, len(VALID_DATASET))
    num_batches = int(len(VALID_DATASET)/BATCH_SIZE)

    total_regression_l2_rel_loss = 0
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_x, batch_y = get_batch(VALID_DATASET, valid_idxs, start_idx, end_idx)
        batch_x, batch_y = TRAIN_DATASET.normalize(batch_x, batch_y)
        bandwidth = BANDWIDTH

        feed_dict = {ops['pointclouds_pl']: batch_x,
                    ops['labels_pl']: batch_y,
                    ops['is_training_pl']: is_training}
        step, loss_val, pred_val = sess.run([ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        total_regression_l2_rel_loss += loss_val
    EPOCH_CNT += 1
    return total_regression_l2_rel_loss / NTEST


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
