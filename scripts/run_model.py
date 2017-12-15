# Imports
import argparse
import tensorflow as tf

from data_load import get_data
from model_estimator import model_estimator
from rnn import rnn
from cnn import cnn
from cnn_rnn_hybrid import cnn_rnn_hybrid as crnn


def main(model_type,
         train_data_dir_path, train_labels_file_path,
         eval_data_dir_path, eval_labels_file_path):

    # Get data
    # ======================================================================
    data_format = 'spec'
    train_data, train_labels = get_data(
        data_dir_path=train_data_dir_path,
        labels_file_path=train_labels_file_path,
        data_format=data_format)

    eval_data, eval_labels = get_data(
        data_dir_path=eval_data_dir_path,
        labels_file_path=eval_labels_file_path,
        data_format=data_format)

    # Get parameters for chosen model
    # ======================================================================
    if model_type == 'cnn':

        params = {'model': cnn,
                  'cnn_wd_lambda': 0.004,
                  'fc_wd_lambda': 0.004,
                  'keep_prob': 0.5,
                  'cnn_num_layers': 2,
                  'conv0_kernel_shape': [5, 5, 64],
                  'conv1_kernel_shape': [5, 5, 64],
                  'conv0_strides': [1, 1, 1, 1],
                  'conv1_strides': [1, 1, 1, 1],
                  'pool0_window_shape': [1, 3, 3, 1],
                  'pool1_window_shape': [1, 3, 3, 1],
                  'pool0_window_strides': [1, 1, 2, 1],
                  'pool1_window_strides': [1, 1, 2, 1],
                  'fc_num_layers': 3,
                  'fc0_n_units': 128,
                  'fc1_n_units': 64,
                  'fc2_n_units': 2}

    elif(model_type == 'rnn'):

        params = {'model': rnn,
                  'rnn_wd_lambda': 0.004,
                  'fc_wd_lambda': 0.004,
                  'keep_prob': 0.5,
                  'rnn_cell': tf.contrib.rnn.GRUCell,
                  'rnn_num_layers': 2,
                  'rnn0_n_units': 10,
                  'rnn1_n_units': 10,
                  'fc_num_layers': 3,
                  'fc0_n_units': 128,
                  'fc1_n_units': 64,
                  'fc2_n_units': 2}

    elif(model_type == 'crnn'):

        params = {'model': crnn,
                  'cnn_wd_lambda': 0.004,
                  'rnn_wd_lambda': 0.004,
                  'fc_wd_lambda': 0.004,
                  'keep_prob': 0.5,
                  'cnn_num_layers': 2,
                  'conv0_kernel_shape': [5, 5, 64],
                  'conv1_kernel_shape': [5, 5, 64],
                  'conv0_strides': [1, 1, 1, 1],
                  'conv1_strides': [1, 1, 1, 1],
                  'pool0_window_shape': [1, 3, 3, 1],
                  'pool1_window_shape': [1, 3, 3, 1],
                  'pool0_window_strides': [1, 2, 2, 1],
                  'pool1_window_strides': [1, 2, 2, 1],
                  'rnn_cell': tf.contrib.rnn.GRUCell,
                  'rnn_num_layers': 2,
                  'rnn0_n_units': 10,
                  'rnn1_n_units': 10,
                  'fc_num_layers': 3,
                  'fc0_n_units': 128,
                  'fc1_n_units': 64,
                  'fc2_n_units': 2}

    # Build, train and evaluate model
    # ======================================================================
    pred = model_estimator(params=params,
                           train_data=train_data,
                           train_labels=train_labels,
                           eval_data=eval_data,
                           eval_labels=eval_labels,
                           test_data=eval_data,
                           train_eval_iterations=100,
                           train_steps=5000,
                           batch_size=128,
                           log_dir_path='../tmp')

    print(pred)


if __name__ == '__main__':

    # Input parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-mt', '--model_type', type=str, dest='model_type',
        choices=['cnn', 'rnn', 'crnn'],
        help='Type of model.')

    parser.add_argument(
        '-tddp', '--train_data_dir_path', type=str, dest='train_data_dir_path',
        help='Path to training data directory.')

    parser.add_argument(
        '-tlfp', '--train_labels_file_path', type=str,
        dest='train_labels_file_path',
        help='Path to training labels file.')

    parser.add_argument(
        '-eddp', '--eval_data_dir_path', type=str, dest='eval_data_dir_path',
        help='Path to evaluation (validation) data directory.')

    parser.add_argument(
        '-elfp', '--eval_labels_file_path', type=str,
        dest='eval_labels_file_path',
        help='Path to training labels file.')

    # Get input arguments
    args = parser.parse_args()

    # Call main function
    main(args.model_type,
         args.train_data_dir_path, args.train_labels_file_path,
         args.eval_data_dir_path, args.eval_labels_file_path)
