# Imports
import numpy as np
import tensorflow as tf
from model_fn import model_fn


# Allow logging to observe model training performance
tf.logging.set_verbosity(tf.logging.INFO)


def model_estimator(params, train_data, train_labels, eval_data,
                    eval_labels, test_data, train_eval_iterations, train_steps,
                    batch_size, log_dir_path):
    """Helper to create a Variable stored on CPU memory.

    Args:
        params - dict. Maps model/estimator parameter names to model/estimator
        parameters.
        train_data - np.array. The training data.
        train_labels - np.array. The training labels.
        train_data - np.array. The evaluation (validation) data.
        train_labels - np.array. The evaluation (validation) labels.
        test_data - np.array. The test data.
        train_eval_iterations - int. Number of training/evaluation
        iterations.
        train_steps - int. Number of training steps per training/evaluation
        iteration.
        batch_size - int. Size of batch for each training step.
        log_dir_path - string. Path to logging directory.

    Returns:
        prediciton - np.array. Model label predictions for test data.

    """

    # Delete directory containing events logs and checkpoints if it exists
    if tf.gfile.Exists(log_dir_path):
        tf.gfile.DeleteRecursively(log_dir_path)

    # Create directory containing events logs and checkpoints
    tf.gfile.MakeDirs(log_dir_path)

    # Setup session configuration
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=0,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    valid_every_n_steps = 50
    config = tf.estimator.RunConfig(
        session_config=sess_config,
        model_dir=log_dir_path,
        save_checkpoints_steps=valid_every_n_steps-1)

    # Create validation monitor
    valid_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=valid_input_fn,
        every_n_steps=valid_every_n_steps)

    # Create the classifier wrapping the model
    params['batch_size'] = batch_size
    params['eval_size'] = eval_data.shape[0]
    params['predict_size'] = test_data.shape[0]
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params)

    val_mon_list = [validation_monitor]
    hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
        val_mon_list,
        classifier)

    # Setup training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)

    # Train the classifier
    classifier.train(
        input_fn=train_input_fn,
        steps=train_steps,
        hooks=hooks)

    # for _ in range(train_eval_iterations):
    #     # Setup training inputs
    #     train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #         x={"x": train_data},
    #         y=train_labels,
    #         batch_size=batch_size,
    #         num_epochs=None,
    #         shuffle=True)

    #     # Train the estimator
    #     classifier.train(
    #         input_fn=train_input_fn,
    #         steps=train_steps)

    #     # Setup evaluation inputs
    #     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #         x={"x": eval_data},
    #         y=eval_labels,
    #         num_epochs=1,
    #         shuffle=False)

    #     # Evaluate the estimator
    #     eval_results = classifier.evaluate(input_fn=eval_input_fn)
    #     print(eval_results)

    # Generate estimator predictions
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      num_epochs=1,
      shuffle=False)

    predictions = np.array(
        list(classifier.predict(input_fn=predict_input_fn))).T

    return predictions
