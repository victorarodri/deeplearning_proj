# Imports
import tensorflow as tf

from data_load import get_data
from model_estimator import model_estimator
# from rnn import rnn, bidirectional_rnn
from cnn import cnn
# from cnn_rnn_hybrid import cnn_rnn_hybrid


# DATA
# ======================================================================
train_data_dir_path = ('../data/spectrogram/training/sub_ws300_full')
train_labels_file_path = ('../data/spectrogram/training/REFERENCE.csv')

eval_data_dir_path = ('../data/spectrogram/validation/sub_ws300_full')
eval_labels_file_path = ('../data/spectrogram/validation/REFERENCE.csv')


data_format = 'spec'
train_data, train_labels = get_data(data_dir_path=train_data_dir_path,
                                    labels_file_path=train_labels_file_path,
                                    data_format=data_format)

eval_data, eval_labels = get_data(data_dir_path=eval_data_dir_path,
                                  labels_file_path=eval_labels_file_path,
                                  data_format=data_format)

# CNN
# ======================================================================
params = {'model': cnn,
          'cnn_wd_lambda': 0.004,
          'fc_wd_lambda': 0.004,
          'keep_prob': 0.5,
          'cnn_num_layers': 4,
          'conv0_kernel_shape': [5, 5, 64],
          'conv1_kernel_shape': [5, 5, 64],
          'conv2_kernel_shape': [5, 5, 64],
          'conv3_kernel_shape': [5, 5, 64],
          # 'conv4_kernel_shape': [5, 5, 64],
          'conv0_strides': [1, 1, 1, 1],
          'conv1_strides': [1, 1, 1, 1],
          'conv2_strides': [1, 1, 1, 1],
          'conv3_strides': [1, 1, 1, 1],
          # 'conv4_strides': [1, 1, 1, 1],
          'pool0_window_shape': [1, 3, 3, 1],
          'pool1_window_shape': [1, 3, 3, 1],
          'pool2_window_shape': [1, 3, 3, 1],
          'pool3_window_shape': [1, 3, 3, 1],
          # 'pool4_window_shape': [1, 3, 3, 1],
          'pool0_window_strides': [1, 1, 2, 1],
          'pool1_window_strides': [1, 1, 2, 1],
          'pool2_window_strides': [1, 1, 2, 1],
          'pool3_window_strides': [1, 1, 2, 1],
          # 'pool4_window_strides': [1, 1, 2, 1],
          'fc_num_layers': 3,
          'fc0_n_units': 128,  # 120
          'fc1_n_units': 64,  # 60
          'fc2_n_units': 2}

# # CNN-RNN hybrid
# # ======================================================================
# params = {'model': cnn_rnn_hybrid,
#           'cnn_wd_lambda': 0.004,
#           'rnn_wd_lambda': 0.004,
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
#           'rnn_cell': tf.nn.rnn_cell.LSTMCell,
#           'rnn_num_layers': 2,
#           'rnn0_n_units': 300,
#           'rnn1_n_units': 100,
#           'fc_num_layers': 2,
#           'fc0_n_units': 50,
#           'fc1_n_units': 2}

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
print(pred.shape)
