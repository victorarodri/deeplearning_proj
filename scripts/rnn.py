# Imports
import tensorflow as tf
from model_var import _variable_on_cpu, _variable_with_weight_decay


def rnn(features, params, mode):
    """Builds architecture for a RNN.

    Args:
        features - dict. Contains data inputs.
            'x' - np.array. Data for input layer.
        params - dict. Contains model parameters
            Must Contain:
                'rnn_cell' - tf.nn.rnn_cell.RNNCell. RNN cell for the RNN.
                'rnn_num_layers' - int. Number of RNN layers.
                'rnnX_n_units' - int. Number of units in RNN layer X.
                'fc_num_layers' - int. Number of FC layers.
                'fcX_n_units' - int. Number of units in FC layer X
                'fc_wd_lambda' - float. Strength of L2 regularization on FC
                layer weights
            May Contain:
                'keep_prob' - float [0,1]. (1 - Probability of dropout) for FC
                layers.
        mode - Current mode: TRAIN, EVAL, or PREDICT.

    Returns:
        unscaled_logits - tensor. Unscaled output from last FC layer.

    """
    # Input Layer
    input_layer = tf.reshape(
        features['x'],
        [-1, features['x'].shape[1].value, features['x'].shape[2].value])

    # Multilayer RNN
    with tf.variable_scope(name_or_scope='rnn') as scope:
        # Pull RNN cell from params
        rnn_cell = params['rnn_cell']

        # Define multilayer RNN cell
        rnn_cell_mult = tf.contrib.rnn.MultiRNNCell(
            [rnn_cell(params['rnn' + str(i) + '_n_units'])
             for i in range(params['rnn_num_layers'])])

        # Define RNN
        outputs, rnn_state = tf.nn.dynamic_rnn(
            cell=rnn_cell_mult,
            inputs=input_layer,
            dtype=tf.float32,
            time_major=False,
            scope=scope.name)

    # Fully connected layers
    fc_features = {'x': outputs}
    unscaled_logits = _rnn_fc_layers(fc_features, params, mode)

    return unscaled_logits


def bidirectional_rnn(features, params, mode):
    """Builds architecture for a bidirectional RNN.

    Args:
        features - dict. Contains data inputs.
            'x' - np.array. Data for input layer
        params - dict. Contains model parameters.
            Must Contain:
                'rnn_fw_cell' - tf.nn.rnn_cell.RNNCell. RNN cell for forward
                RNN.
                'rnn_bw_cell' - tf.nn.rnn_cell.RNNCell. RNN cell for backward
                RNN.
                'rnn_num_layers' - int. Number of RNN layers.
                'rnnX_fw_n_units' - int. Number of units in forward RNN cell
                in layer X.
                'rnnX_bw_n_units' - int. Number of units in backward RNN cell
                in layer X.
                'fc_num_layers' - int. Number of FC layers.
                'fcX_n_units' - int. Number of units in FC layer X
                'fc_wd_lambda' - float. Strength of L2 regularization on FC
                layer weights
            May Contain:
                'keep_prob' - float [0,1]. (1 - Probability of dropout) for FC
                layers.
        mode - Current mode: TRAIN, EVAL, or PREDICT.

    Returns:
        unscaled_logits - tensor. Unscaled output from last FC layer.
    """

    # Input Layer
    input_layer = tf.reshape(
        features['x'],
        [-1, features['x'].shape[1].value, features['x'].shape[2].value])

    # Multilayer RNN
    with tf.variable_scope(name_or_scope='rnn') as scope:
        # Pull RNN cells from params
        rnn_fw_cell = params['rnn_fw_cell']
        rnn_bw_cell = params['rnn_bw_cell']

        # Define multilayer forward and backward RNN cells
        rnn_fw_cell_mult = tf.contrib.rnn.MultiRNNCell(
            [rnn_fw_cell(params['rnn' + str(i) + '_fw_n_units'])
             for i in range(params['rnn_num_layers'])])

        rnn_bw_cell_mult = tf.contrib.rnn.MultiRNNCell(
            [rnn_bw_cell(params['rnn' + str(i) + '_bw_n_units'])
             for i in range(params['rnn_num_layers'])])

        # Define bidirectional RNN
        outputs, rnn_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_fw_cell_mult,
            cell_bw=rnn_bw_cell_mult,
            inputs=input_layer,
            dtype=tf.float32,
            time_major=False,
            scope=scope.name)

        # Concatenate outputs from fw and bw RNNs
        outputs = tf.concat(outputs, 2)

    # Fully connected layers
    fc_features = {'x': outputs}
    unscaled_logits = _rnn_fc_layers(fc_features, params, mode)

    return unscaled_logits


def _rnn_fc_layers(features, params, mode):
    """Builds fully connected layers for RNN architectures.

    Args:
        features - dict. Contains data inputs.
            'x' - np.array. Data for input layer (RNN output)
        params - dict. Contains model parameters.
            Relevant keys are:
            'fc_num_layers' - int. Number of FC layers.
            'fcX_n_units' - int. Number of units in FC layer X
            'fc_wd_lambda' - float. Strength of L2 regularization on weights.
            'keep_prob' - float [0,1]. 1 - Probability of dropout
        mode - Current mode: TRAIN, EVAL, or PREDICT.

    Returns:
        unscaled_logits - tensor. Unscaled output from last FC layer.
    """

    # Pull inputs from features
    inputs = features['x']

    # Define dictionary for tracking FC layer inputs
    fc_layer_inputs = {'fc0': inputs}

    # Fully connected layers
    for i in range(params['fc_num_layers']):
        with tf.variable_scope(name_or_scope='fc' + str(i)) as scope:

            # Define layer weights with weight decay
            weights = _variable_with_weight_decay(
                name='weights',
                shape=[fc_layer_inputs['fc' + str(i)].shape[-1],
                       params['fc' + str(i) + '_n_units']],
                stddev=0.04,
                wd=params['fc_wd_lambda'])

            # Define layer bias
            bias = _variable_on_cpu(
                name='bias',
                shape=[params['fc' + str(i) + '_n_units']],
                initializer=tf.constant_initializer(0.1))

            # Get inputs for current layer
            inputs = fc_layer_inputs['fc' + str(i)]

            # Apply weights and biases to layer inputs
            if i == 0:
                # Pass RNN output from last cell as input to FC0
                pre_activation = tf.add(
                    tf.matmul(inputs[:, -1, :], weights), bias)
            else:
                # Pass FC(i-1) output as input to FCi
                pre_activation = tf.add(
                    tf.matmul(inputs, weights), bias)

            # Apply nonliner activation function
            outputs = tf.nn.relu(
                features=pre_activation,
                name=scope.name)

            # Apply dropout if mode is TRAIN
            if ('keep_prob' in params.keys() and
                    mode == tf.estimator.ModeKeys.TRAIN and
                    i < params['fc_num_layers'] - 1):

                fc_layer_inputs['fc' + str(i + 1)] = tf.nn.dropout(
                    outputs,
                    keep_prob=params['keep_prob'])

            else:
                fc_layer_inputs['fc' + str(i + 1)] = outputs

    # Naming network output for clarity
    unscaled_logits = fc_layer_inputs['fc' + str(params['fc_num_layers'])]

    return unscaled_logits
