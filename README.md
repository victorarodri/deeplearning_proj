# Classification of Normal versus Abnormal Heart Sounds 
Our goal is to classify heart sound recordings, or phonocardiograms (PCGs) as either normal (i.e. not suggestive of cardiac pathology) or abnormal (i.e. suggetive of possible cardiac pathology). To achieve this goal we train deep neural networks including convolutional (CNN), recurrent (RNN) and combined CNN + RNN (CRNN) architectures to function as binary PCG classifiers. Data for this project was obtained from the [2016 Computing in Cardiology dataset](https://www.physionet.org/challenge/2016/	).

## Getting Started

### Prerequisites

This repository requires the following dependencies.

```
numpy==1.13.3
scipy==1.0.0
six==1.11.0
tensorflow==1.4.1
tensorflow-gpu==1.4.1
```

To install these simply clone this repository, change to its root directory, and run the following code in your terminal command line.

```
pip install -r requirements.txt
```

## Running on sample data

A small amount of training and validation data (PCG spectrograms) has been included in this repository. You can use these data to train models described in `cnn.py`, `rnn.py`, and `cnn_rnn_hybrid.py` (a CRNN) which can be found in the `scripts` directory. Change directories to `scripts` and run the following command to train a CNN.

```
python run_model.py -mt cnn -tddp '../data/spectrogram/training/sub_ws300' -tlfp '../data/spectrogram/training/REFERENCE.csv' -eddp '../data/spectrogram/validation/sub_ws300' -elfp '../data/spectrogram/validation/REFERENCE.csv'
```

To run the RNN or CRNN replace `-mt cnn` with `-mt rnn` or `-mt crnn` respectively.