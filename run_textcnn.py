# coding:utf-8\
import argparse
import logging
import sys
import os
import yaml

import tensorflow as tf
import numpy as np

from textcnn import TextCNN
path = os.path.abspath(os.path.join(os.path.dirname("__file__"),
                                    os.path.pardir))
sys.path.append(path)
from utils import loading_dict
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Loading parameters for textcnn")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info("Initializing fresh parameters for textcnn")

def train_model(sess, model, epochs=10):
    """training model"""
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)
    num_batches = len(model.y) / model.batch_size + 1
    num_samples = len(y_train)
    for e in range(epochs):
        losses = []
        for i in range(num_batches):
            X_batch = X_train[i*model.batch_size:(i+1)*model.batch_size]
            y_batch = y_train[i*model.batch_size:(i+1)*model.batch_size]
            feed_dict = {model.X: X_batch, model.y:y_batch}
            loss, summary, global_step, _ = sess.run([model.loss, merged, model.global_step,
                                                      model.train_op], feed_dict=feed_dict)
            # aggregate performance stats
            actual_batch_size = len(y_batch)
            losses.append(loss*actual_batch_size)
            train_writer.add_summary(summary, global_step=global_step)
            if global_step % 500 == 0:
                logging.info('Iteration {0}: with mini-batch taining loss = {1:.2f}'
                             .format(global_step, loss))
        # loss of each epoch
        total_loss = np.sum(losses) / num_samples
        validation_loss = sess.run(model.loss, feed_dict={model.X:X_valid, model.y:y_valid})
        saver.save(sess, 'checkpoints/model', global_step=global_step)
        logging.info('Epoch {0}, Overall training loss = {1:.2f}, validation loss = {2:.2f}'
                     .format(e+1, total_loss, validation_loss))

def test_model(sess, model):
    num_batches = len(y_test) / model.batch_size + 1
    num_samples = len(y_test)
    losses = []
    for i in num_batches:
        X_batch = X_test[i*model.batch_size:(i+1)*model.batch_size]
        y_batch = y_test[i*model.batch_size:(i+1)*model.batch_size]
        feed_dict = {model.X: X_batch, model.y:y_batch}
        loss = sess.run(model.loss, feed_dict=feed_dict)
        # aggregate performance stats
        actual_batch_size = len(y_batch)
        losses.append(loss*actual_batch_size)
        if i % 100 == 0:
            logging.info('batch index {0}: with mini-batch test loss = {1:.2f}'.format(i, loss))
    total_loss = np.sum(losses) / num_samples
    logging.info('Overall testing loss {0:.2f}'.format(total_loss))

if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # get mode (train or test)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train or test', type=str)
    args = parser.parse_args()
    mode = args.mode
    # configuration
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 128
    config['num_classes'] = 39
    config['embedding_size'] = 300
    config['keep_prob'] = 1.0
    config['filter_sizes'] = [7,8,9]
    config['num_filters'] = 300
    config['sentence_length'] = 2500
    # init data path
    train_data_path = '../../corpus/newdata.clean.dat'
    test_data_path =  '../../corpus/stdtestSet.dat'
    channel2id_path =  '../../corpus/channel2cid.yaml'
    cid2channel_path = '../../corpus/cid2channel.yaml'
    dict_path = '../../corpus/dict_texts'
    # loading data
    X_train = np.load()
    y_train = np.load()
    # build model
    model = TextCNN(config)
    model.build_graph()
    # running model
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        check_restore_parameters(sess, saver)
        if mode == 'train':
            print('starting training...')
            train_model(sess, model, epochs=20)
        if mode =='test':
            print('start testing...')
            test_model(sess, model)
