# Imports
import tensorflow as tf
from model_var import _variable_on_cpu, _variable_with_weight_decay
from rnn import _rnn_fc_layers


def cnn_rnn_hybrid(features, params, mode):
    """Builds architecture for a CNN-RNN hybrid.

    Args:
        features - dict. Contains data inputs.
            'x' - np.array. Data for input layer.
        params - dict. Contains model parameters
            Must Contain:
                'cnn_num_layers' - int. Number of CNN layers.
                'convX_kernel_shape' - array like. Shape of convolution kernel
                in layer X.
                'convX_strides' - array like. Strides for kernel in layer X.
                'poolX_window_shape' - array like. Shape of pooling window.
                'poolX_window_strides' - array like Stride for window in
                layer X.
                'cnn_wd_lambda' - float. Strength of L2 regularization on
                convolutional kernel weights.
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
    # Input Layer.
    input_layer = features['x']
    input_layer = tf.reshape(
        input_layer,
        [-1, input_layer.shape[1].value, input_layer.shape[2].value, 1])

    cnn_layer_inputs = {'cnn0': input_layer}

    # Convolutional layers
    for i in range(params['cnn_num_layers']):
        # Convolution layer
        with tf.variable_scope(name_or_scope='conv' + str(i)) as scope:
            # Pull kernel shape from params
            kernel_shape = params['conv' + str(i) + '_kernel_shape']

            # Add number of input channels to kernel
            kernel_shape.extend(
                [cnn_layer_inputs['cnn' + str(i)].shape[-1].value])

            # Reshape kernel to proper format
            kernel_shape[2], kernel_shape[3] = kernel_shape[3], kernel_shape[2]

            kernel = _variable_with_weight_decay(
                name='weights',
                shape=kernel_shape,
                stddev=5e-2,
                wd=params['cnn_wd_lambda'])

            conv = tf.nn.conv2d(
                input=cnn_layer_inputs['cnn' + str(i)],
                filter=kernel,
                strides=params['conv' + str(i) + '_strides'],
                padding='SAME')

            biases = _variable_on_cpu(
                name='biases',
                shape=[kernel_shape[3]],
                initializer=tf.constant_initializer(0.0))

            pre_activation = tf.nn.bias_add(conv, biases)

            # cnn_layer_outputs = tf.nn.tanh(
            #     x=pre_activation,
            #     name=scope.name)
            cnn_layer_outputs = tf.nn.relu(
                features=pre_activation,
                name=scope.name)

        # Max pooling layer
        pool_layer_outputs = tf.nn.max_pool(
            value=cnn_layer_outputs,
            ksize=params['pool' + str(i) + '_window_shape'],
            strides=params['pool' + str(i) + '_window_strides'],
            padding='SAME',
            name='pool' + str(i))

        # Local response normalization layer
        norm_layer_outputs = tf.nn.lrn(
            input=pool_layer_outputs,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm' + str(i))

        cnn_layer_inputs['cnn' + str(i + 1)] = norm_layer_outputs

    # Reshape inputs to RNN
    rnn_input = cnn_layer_inputs['cnn' + str(params['cnn_num_layers'])]
    rnn_input = tf.transpose(rnn_input, perm=[0, 2, 1, 3])
    rnn_input = tf.reshape(
        rnn_input,
        [-1, rnn_input.shape[1].value,
         rnn_input.shape[2].value * rnn_input.shape[3].value])

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
            inputs=rnn_input,
            dtype=tf.float32,
            time_major=False,
            scope=scope.name)

    # Fully connected layers
    fc_features = {'x': outputs}
    unscaled_logits = _rnn_fc_layers(fc_features, params, mode)

    return unscaled_logits
