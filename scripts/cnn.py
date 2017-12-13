# Imports
import tensorflow as tf
from model_var import _variable_on_cpu, _variable_with_weight_decay


def cnn(features, params, mode):
    """Builds architecture for a CNN.

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
    # Input Layer
    input_layer = tf.transpose(features['x'], perm=[0, 2, 1])
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

            cnn_layer_outputs = tf.nn.tanh(
                x=pre_activation,
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

    # Fully connected layers
    fc_features = {'x': cnn_layer_inputs['cnn' + str(params['cnn_num_layers'])]}

    unscaled_logits = _cnn_fc_layers(fc_features, params, mode)

    return unscaled_logits


def _cnn_fc_layers(features, params, mode):
    """Builds fully connected layers for CNN architectures.

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

    # Pull inputs from features and flatten
    inputs = features['x']

    print('FC inputs shape: {}'.format(inputs.shape))

    inputs_flat_shape = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
    if mode == tf.estimator.ModeKeys.TRAIN:
        inputs = tf.reshape(inputs, [params['batch_size'], inputs_flat_shape])
    elif mode == tf.estimator.ModeKeys.EVAL:
        if params['eval_size'] > 128:
            inputs = tf.reshape(inputs, [128, inputs_flat_shape])
        else:
            inputs = tf.reshape(
                inputs, [params['eval_size'], inputs_flat_shape])
    elif mode == tf.estimator.ModeKeys.PREDICT:
        inputs = tf.reshape(
            inputs, [params['predict_size'], inputs_flat_shape])

    print('FC flattened inputs shape: {}'.format(inputs.shape))

    # Define dictionary for tracking FC layer inputs
    fc_layer_inputs = {'fc0': inputs}

    # Fully connected layers
    for i in range(params['fc_num_layers']):
        with tf.variable_scope(name_or_scope='fc' + str(i)) as scope:

            # Define layer weights with weight decay
            if i == 0:
                weights = _variable_with_weight_decay(
                    name='weights',
                    shape=[fc_layer_inputs['fc' + str(i)].shape[-1].value,
                           params['fc' + str(i) + '_n_units']],
                    stddev=0.04,
                    wd=params['fc_wd_lambda'])
            else:
                weights = _variable_with_weight_decay(
                    name='weights',
                    shape=[fc_layer_inputs['fc' + str(i)].shape[-1].value,
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
            pre_activation = tf.add(tf.matmul(inputs, weights), bias)

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
