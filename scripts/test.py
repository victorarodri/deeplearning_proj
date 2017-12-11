# Imports
import tensorflow as tf

from data_load import get_data
from model_estimator import model_estimator
from rnn import rnn, bidirectional_rnn
from cnn import cnn
from cnn_rnn_hybrid import cnn_rnn_hybrid


# DATA
# ======================================================================
data_dir_path = ('/Volumes/light/deeplearning_proj/data/'
                 'spectrogram/training/sub_ws300')

labels_file_path = ('/Volumes/light/deeplearning_proj/data/'
                    'spectrogram/training/REFERENCE.csv')

data_format = 'spec'
sample_size = 12
data, labels = get_data(data_dir_path=data_dir_path,
                        labels_file_path=labels_file_path,
                        data_format=data_format,
                        sample_size=sample_size)

# # CNN
# # ======================================================================
# params = {'model': cnn,
#           'cnn_wd_lambda': 0.004,
#           'fc_wd_lambda': 0.004,
#           'keep_prob': 0.5,
#           'cnn_num_layers': 2,
#           'conv0_kernel_shape': [5, 5, 64],
#           'conv1_kernel_shape': [5, 5, 64],
#           'conv0_strides': [1, 1, 1, 1],
#           'conv1_strides': [1, 1, 1, 1],
#           'pool0_window_shape': [1, 3, 3, 1],
#           'pool1_window_shape': [1, 3, 3, 1],
#           'pool0_window_strides': [1, 2, 2, 1],
#           'pool1_window_strides': [1, 2, 2, 1],
#           'fc_num_layers': 2,
#           'fc0_n_units': 7,
#           'fc1_n_units': 2}

# pred = model_estimator(params=params,
#                        train_data=data,
#                        train_labels=labels,
#                        eval_data=data,
#                        eval_labels=labels,
#                        test_data=data,
#                        train_eval_iterations=2,
#                        train_steps=2,
#                        batch_size=2,
#                        log_dir_path='../tmp')

# # RNN
# # ======================================================================
# params = {'model': rnn,
#           'rnn_wd_lambda': 0.004,
#           'fc_wd_lambda': 0.004,
#           'keep_prob': 0.5,
#           'rnn_cell': tf.nn.rnn_cell.LSTMCell,
#           'rnn_num_layers': 2,
#           'rnn0_n_units': 10,
#           'rnn1_n_units': 10,
#           'fc_num_layers': 2,
#           'fc0_n_units': 7,
#           'fc1_n_units': 2}

# pred = model_estimator(params=params,
#                        train_data=data,
#                        train_labels=labels,
#                        eval_data=data,
#                        eval_labels=labels,
#                        test_data=data,
#                        train_eval_iterations=2,
#                        train_steps=2,
#                        batch_size=2,
#                        log_dir_path='../tmp')

# BIDIRECTIONAL RNN
# ======================================================================
# params = {'model': bidirectional_rnn,
#           'rnn_wd_lambda': 0.004,
#           'fc_wd_lambda': 0.004,
#           'keep_prob': 0.5,
#           'rnn_num_layers': 2,
#           'rnn_fw_cell': tf.nn.rnn_cell.LSTMCell,
#           'rnn_bw_cell': tf.nn.rnn_cell.LSTMCell,
#           'rnn0_fw_n_units': 10,
#           'rnn1_fw_n_units': 10,
#           'rnn0_bw_n_units': 10,
#           'rnn1_bw_n_units': 10,
#           'fc_num_layers': 2,
#           'fc0_n_units': 7,
#           'fc1_n_units': 2}

# pred = model_estimator(params=params,
#                        train_data=data,
#                        train_labels=labels,
#                        eval_data=data,
#                        eval_labels=labels,
#                        test_data=data,
#                        train_eval_iterations=2,
#                        train_steps=2,
#                        batch_size=2,
#                        log_dir_path='../tmp')

# CNN-RNN hybrid
# ======================================================================
params = {'model': cnn_rnn_hybrid,
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
          'rnn_cell': tf.nn.rnn_cell.LSTMCell,
          'rnn_num_layers': 2,
          'rnn0_n_units': 10,
          'rnn1_n_units': 10,
          'fc_num_layers': 2,
          'fc0_n_units': 7,
          'fc1_n_units': 2}

pred = model_estimator(params=params,
                       train_data=data,
                       train_labels=labels,
                       eval_data=data,
                       eval_labels=labels,
                       test_data=data,
                       train_eval_iterations=2,
                       train_steps=2,
                       batch_size=2,
                       log_dir_path='../tmp')

print(pred)
print(pred.shape)
