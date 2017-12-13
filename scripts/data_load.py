# Imports
import numpy as np
import glob
import csv
import random
import os


def get_data(data_dir_path, labels_file_path, data_format='td',
             sample_size=None):
    """Helper function for loading heart sound recording data into memory.

    Args:
        data_dir_path - string. Path to data directory.
        labels_file_path - string. Path to label file.
        data_format - string. Type of data format:
            'td' - Time domain sequence.
            'spec' - Power spectrogram.
            'mel' - Mel frequency cepstral coefficients (not yet implemented).
        sample_size - int. Number of files to sample. If None then all files
        are sampled.

    Returns:
        data - np.array. Array of shape [sample_size * max_sequence_length]
        containing loaded sequence data.
        labels - np.array. Array of shape [sample_size] containing sequence
        labels.

    """
    # Load data file names and shuffle them randomly
    data_filenames = glob.glob(os.path.join(data_dir_path, '*npy'))
    random.shuffle(data_filenames)

    # If no sample size specified, set sample_size equal to the total
    # number of data files
    if sample_size is None:
        sample_size = len(data_filenames)

    # Determine shape of data and load
    max_length = 0
    if data_format == 'td':
        for i in range(sample_size):
            df = data_filenames[i]
            max_length = max(max_length, np.load(df).shape[0])

        data = np.zeros([sample_size, max_length])

    # elif data_format == 'spec':
    #     max_width = 0
    #     for i in range(sample_size):
    #         df = data_filenames[i]
    #         max_width = max(max_width, np.load(df).shape[0])
    #         max_length = max(max_length, np.load(df).shape[1])

    #     data = np.zeros([sample_size, max_length, max_width])

    if data_dir_path.split('_')[-1] == 'ws300':
        data = np.zeros([sample_size, 100, 151])
    elif data_dir_path.split('_')[-1] == 'ws500':
        data = np.zeros([sample_size, 100, 251])
    elif data_dir_path.split('_')[-1] == 'ws1000':
        data = np.zeros([sample_size, 100, 501])

    # Load dictionary mapping file name prefix to label
    labels_dict = __get_labels_dict(labels_file_path)

    labels = np.zeros(sample_size)
    for i in range(sample_size):
        df = data_filenames[i]
        if data_format == 'td':
            data[i, :] = np.load(df)
        elif data_format == 'spec':
            # Data are formated Freq x Time.
            # Tranpose to get Time X Freq which complies with
            # time_major = False format for RNNs
            data[i, :, :] = np.load(df).T

        label_key = df.split('/')[-1].split('.')[0].split('_')[0]
        labels[i] = labels_dict[label_key]

    return data.astype(np.float32), labels.astype(np.float32)


def __get_labels_dict(labels_file_path):
    """Helper function for loading heart sound recording data into memory.

    Args:
        labels_file_path - string. Path to label file.

    Returns:
        labels_dict - dict. Dicitionary mapping data file name prefixes to
        labels.

    """
    labels_dict = {}
    with open(labels_file_path) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            # Convert -1 labels to 0 (tensorflow expects non-negative,
            # sequential labels)
            if row[1] == '-1':
                label = 0
            else:
                label = 1

            labels_dict[row[0]] = label

    return labels_dict
