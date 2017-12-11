# Imports
import tensorflow as tf


def model_fn(features, labels, mode, params, config=None):
    """Function for training, evaluating and making predictions with model.
    Args:
        features - dict. Features to pass to model.
        labels - array like. Labels for data in features.
        mode - Current mode: TRAIN, EVAL, or PREDICT.
        params - Parameters to pass to model.
            Must contain:
                'model' - function. Function defining model architecture.
            May contain:
                'rnn_wd_lambda' - float. Weight decay strength for RNN kernels.
        config - Configuration (not yet implemented).

    Returns:
        tf.EstimatorSpec containing output of current mode.

    """
    # Pull model function from params
    model = params['model']
    unscaled_logits = model(features, params, mode)

    # Generate predictions for PREDICT and EVAL modes.
    predictions = {
        'classes': tf.argmax(input=unscaled_logits, axis=1),
        'probs': tf.nn.softmax(unscaled_logits, name='softmax_tensor')
    }

    # PREDICT
    # ======================================================================
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions['probs'])

    else:
        # Calculate total loss for both TRAIN and EVAL modes.
        labels = tf.cast(labels, tf.int64)
        if 'rnn_wd_lambda' in params.keys():
            total_loss = loss(unscaled_logits, labels, params['rnn_wd_lambda'])
        else:
            total_loss = loss(unscaled_logits, labels)

        # Add summary operation for total loss visualizaiton.
        tf.summary.scalar(
            name='total_loss',
            tensor=total_loss)

        # TRAIN
        # ======================================================================
        if mode == tf.estimator.ModeKeys.TRAIN:

            # Compute gradients using Gradient Descent Optimizer.
            optimizer = tf.train.AdamOptimizer()

            grads_vars = optimizer.compute_gradients(loss=total_loss)

            # Add summary operations for gradient visualizations.
            for grad, var in grads_vars:
                if grad is not None:
                    tf.summary.histogram(name=var.op.name + '/gradients',
                                         values=grad)

            train_op = optimizer.minimize(
                loss=total_loss,
                global_step=tf.train.get_global_step())

            # Add evaluation metrics for TRAIN mode.
            accuracy_train = tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"])

            # Add summary operation for training accuracy visualizaiton.
            tf.summary.scalar(name='accuracy_train',
                              tensor=accuracy_train[0])

            train_summary_hook = tf.train.SummarySaverHook(
                save_steps=10,
                output_dir='models/rnn',
                summary_op=tf.summary.merge_all())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[train_summary_hook])

        # EVALUATE
        # ======================================================================
        else:
            accuracy_valid = tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"])

            # Add summary operation for validation accuracy visualizaiton.
            tf.summary.scalar(
                name='accuracy_validation',
                tensor=accuracy_valid[0])

            eval_metric_ops = {"accuracy": accuracy_valid}

            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=1,
                output_dir='models/rnn',
                summary_op=tf.summary.merge_all())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metric_ops,
                training_hooks=[eval_summary_hook])


def loss(unscaled_logits, labels, rnn_wd_lambda=None):
    """Add L2 loss to all the trainable variables.

    Args:
        unscaled_logits - array like. Output from model for given batch.
        labels - array like. Labels for examples in batch.
        rnn_wd_lambda - float. Strength of weight decay.

    Returns:
        total_loss - float. Total loss for batch.

    """
    # Calculate cross entropy mean and add to losses collection
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=unscaled_logits,
        name='cross_entropy')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection(name='losses', value=cross_entropy_mean)

    # Apply L2 regularization to RNN kernels and add to losses collection
    # NOTE: Kernels/weights for CNN/FC layers are added to losses collection
    # upon creation)
    if rnn_wd_lambda is not None:
        for var in tf.trainable_variables():
            var_scope = var.name.split('/')[0]
            var_type = var.name.split('/')[-1].split(':')[0]

            if var_scope == 'rnn' and var_type == 'kernel':
                tf.add_to_collection(
                    name='losses',
                    value=tf.multiply(tf.nn.l2_loss(var), rnn_wd_lambda))

    # Calculate total loss
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss
