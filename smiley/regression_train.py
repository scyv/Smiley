import configparser
import os
import numpy
import prepare_training_data
import utils
import regression_model
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError


def train():
    print("\nSOFTMAX REGRESSION TRAINING STARTED.")

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

    MODEL_PATH = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['MODELS'],
                              config['DEFAULT']['IMAGE_SIZE'], config['REGRESSION']['MODEL_FILENAME'])
    IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])
    BATCH_SIZE = int(config['DEFAULT']['TRAIN_BATCH_SIZE'])

    # get training/validation/testing data
    try:
        curr_number_of_categories, train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = prepare_training_data.prepare_data(
            "regression", True)
    except Exception as inst:
        raise Exception(inst.args[0])

    # regression model
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE], name="image")  # regression input placeholder
    y_ = tf.placeholder(tf.float32, [None, curr_number_of_categories], name="labels")  # regression ground truth labels
    y, variables = regression_model.regression(x, nCategories=curr_number_of_categories)  # regression output and variables

    # training variables
    with tf.name_scope("Loss"):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    with tf.name_scope("GradientDescent"):
        train_step = tf.train.GradientDescentOptimizer(float(config['REGRESSION']['LEARNING_RATE'])).minimize(
            cross_entropy)
    with tf.name_scope("Acc"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0)

    # merge training data and validation data
    validation_total_data = numpy.concatenate((validation_data, validation_labels), axis=1)
    new_train_total_data = numpy.concatenate((train_total_data, validation_total_data))
    train_size = new_train_total_data.shape[0]

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(variables)

    # training cycle (number of batches and epochs)
    total_batch = int(train_size / BATCH_SIZE)
    epochs = int(config['REGRESSION']['EPOCHS'])

    # restore stored regression model if it exists and has the correct number of categories
    max_acc = maybe_restore_model(MODEL_PATH, saver, sess, accuracy, validation_data, x, validation_labels, y_)

    # loop for epoch
    for epoch in range(epochs):

        # random shuffling
        numpy.random.shuffle(train_total_data)
        train_data_ = new_train_total_data[:, :-curr_number_of_categories]
        train_labels_ = new_train_total_data[:, -curr_number_of_categories:]

        # loop over all batches
        for i in range(total_batch):
            # compute the offset of the current minibatch in the data.
            offset = (i * BATCH_SIZE) % train_size
            batch_xs = train_data_[offset:(offset + BATCH_SIZE), :]
            batch_ys = train_labels_[offset:(offset + BATCH_SIZE), :]

            _, train_accuracy = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})

            # update progress
            progress = float((epoch * total_batch + i + 1) / (epochs * total_batch))
            utils.update_progress(progress)

            validation_accuracy = compute_accuracy(sess, accuracy, train_accuracy, i, total_batch, epoch,
                                                   validation_data, x,
                                                   validation_labels, y_,
                                                   int(config['LOGS']['TRAIN_ACCURACY_DISPLAY_STEP']),
                                                   int(config['LOGS']['VALIDATION_STEP']))

            # save the current model if the maximum accuracy is updated
            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_PATH, write_meta_graph=False, write_state=False)
                print("Model updated and saved in file: %s" % save_path)

            # break inner loop if stop training is required
            if utils.train_should_stop():
                break

        # break outer loop if stop training is required
        if utils.train_should_stop():
            break

    # Code with test set
    # restore variables from disk
    # saver.restore(sess, MODEL_PATH)

    # Code with test set
    # calculate accuracy for all test images
    #test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
    #print("test accuracy for the stored model: %g" % test_accuracy)

    sess.close()

    print("SOFTMAX REGRESSION TRAINING END.")


def maybe_restore_model(model_path, saver, sess, accuracy, validation_data, x, validation_labels, y_):
    exists = os.path.isfile(model_path)
    if not exists:
        return 0

    try:
        saver.restore(sess, model_path)
        # save the current maximum accuracy value for validation data
        max_acc = sess.run(accuracy, feed_dict={x: validation_data, y_: validation_labels})
    except:
        # initialize the maximum accuracy value for validation data
        max_acc = 0.
    return max_acc


def compute_accuracy(sess, accuracy, train_accuracy, i, total_batch, epoch, validation_data, x, validation_labels, y_,
                     DISPLAY_STEP, VALIDATION_STEP):
    if i % DISPLAY_STEP == 0:
        print("Epoch:", '%04d,' % (epoch + 1),
              "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

    # get accuracy for validation data
    validation_accuracy = 0
    if i % VALIDATION_STEP == 0:
        # calculate accuracy
        validation_accuracy = sess.run(accuracy, feed_dict={x: validation_data, y_: validation_labels})
        print("Epoch:", '%04d,' % (epoch + 1),
              "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

    return validation_accuracy


if __name__ == '__main__':
    train()
