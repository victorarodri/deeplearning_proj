# Imports
import tensorflow as tf


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name - string. Variable name.
        shape - array like.  Variable shape.
        initializer - array like. Variable initializer.

    Returns:
        var - Tensorflow variable.

    """
    with tf.device('/gpu:0'):
        var = tf.get_variable(name=name,
                              shape=shape,
                              dtype=tf.float32,
                              initializer=initializer)

    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Args:
        name - string. Variable name.
        shape - array like.  Variable shape.
        stddev - float. Standard deviation parameter for initializer
        wd - float. weight decay rate.

    Returns:
        var - Tensorflow variable with weight decay.

    """
    initializer = tf.truncated_normal_initializer(stddev=stddev,
                                                  dtype=tf.float32)

    var = _variable_on_cpu(name=name,
                           shape=shape,
                           initializer=initializer)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name='weight_loss')

        tf.add_to_collection('losses', weight_decay)

    return var
