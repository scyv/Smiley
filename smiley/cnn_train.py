import configparser
import os
import cnn_model
import numpy
import prepare_training_data
import utils
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError


def train():
    print("\nCNN TRAINING STARTED.")

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

    MODEL_PATH = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['MODELS'],
                              config['DEFAULT']['IMAGE_SIZE'], config['CNN']['MODEL_FILENAME'])
    IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])
    BATCH_SIZE = int(config['DEFAULT']['TRAIN_BATCH_SIZE'])
    LOGS_DIRECTORY = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['LOGS'])

    # get training/validation/testing data
    try:
        curr_number_of_categories, train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = prepare_training_data.prepare_data(
            "CNN", True)
    except Exception as inst:
        raise Exception(inst.args[0])

    # CNN model
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE], name="image")  # CNN input placeholder
    y_ = tf.placeholder(tf.float32, [None, curr_number_of_categories], name="labels")  # CNN ground truth labels
    y, variables = cnn_model.convolutional(x, nCategories=curr_number_of_categories)  # CNN output and variables

    is_training = tf.placeholder(tf.bool)  # used for activating the dropout in training phase

    # loss function
    with tf.name_scope("Loss"):
        loss = tf.losses.softmax_cross_entropy(y_, y)

    # create a summary to monitor loss tensor
    tf.summary.scalar('loss', loss)

    # define optimizer
    with tf.name_scope("ADAM"):
        # optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
            float(config['CNN']['LEARNING_RATE']),  # base learning rate
            tf.cast(batch, dtype=tf.int16) * BATCH_SIZE,  # current index in the dataset
            train_size,  # decay step
            0.95,  # decay rate
            staircase=True)

        # use simple momentum for the optimization
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

    # create a summary to monitor learning_rate tensor
    tf.summary.scalar('learning_rate', learning_rate)

    # get accuracy of model
    with tf.name_scope("Acc"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0)

    # create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})
    saver = tf.train.Saver(variables)

    # training cycle (number of batches and epochs)
    total_batch = int(train_size / BATCH_SIZE)
    epochs = int(config['CNN']['EPOCHS'])

    # op to write logs to Tensorboard
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # restore stored CNN model if it exists and has the correct number of categories
    max_acc = maybe_restore_model(MODEL_PATH, saver, sess, accuracy, validation_data, x, validation_labels, y_,
                                  is_training)

    # loop for epoch
    for epoch in range(epochs):

        # random shuffling
        numpy.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-curr_number_of_categories]
        train_labels_ = train_total_data[:, -curr_number_of_categories:]

        # loop over all batches
        for i in range(total_batch):
            # compute the offset of the current minibatch in the data.
            offset = (i * BATCH_SIZE) % train_size
            batch_xs = train_data_[offset:(offset + BATCH_SIZE), :]
            batch_ys = train_labels_[offset:(offset + BATCH_SIZE), :]

            # run optimization op (backprop), loss op (to get loss value) and summary nodes
            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op],
                                                  feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # update progress
            progress = float((epoch * total_batch + i + 1) / (epochs * total_batch))
            utils.update_progress(progress)

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            validation_accuracy = compute_accuracy(sess, accuracy, train_accuracy, i, total_batch, epoch,
                                                   validation_data, x,
                                                   validation_labels, y_, is_training,
                                                   int(config['LOGS']['TRAIN_ACCURACY_DISPLAY_STEP']),
                                                   int(config['LOGS']['VALIDATION_STEP']))

            # save the current model if the maximum accuracy is updated
            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_PATH, write_meta_graph=False, write_state=False)
                print("Model updated and saved in file: %s" % save_path)

                # saver.save(sess, LOGS_DIRECTORY + "CNN", epoch)

            # break inner loop if stop training is required
            if utils.train_should_stop():
                break;

        # break outer loop if stop training is required
        if utils.train_should_stop():
            break;

    # Code with test set
    # restore variables from disk
    # saver.restore(sess, MODEL_PATH)

    # Code with test set
    # calculate accuracy for all test images
    #test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels, is_training: False})
    #print("test accuracy for the stored model: %g" % test_accuracy)

    sess.close()

    print("CNN TRAINING END.")


def maybe_restore_model(model_path, saver, sess, accuracy, validation_data, x, validation_labels, y_, is_training):
    exists = os.path.isfile(model_path)
    if not exists:
        return 0

    try:
        saver.restore(sess, model_path)
        # save the current maximum accuracy value for validation data
        max_acc = sess.run(accuracy,
                           feed_dict={x: validation_data, y_: validation_labels,
                                      is_training: False})
    except:
        # initialize the maximum accuracy value for validation data
        max_acc = 0.
    return max_acc


def compute_accuracy(sess, accuracy, train_accuracy, i, total_batch, epoch, validation_data, x, validation_labels, y_,
                     is_training, DISPLAY_STEP, VALIDATION_STEP):
    if i % DISPLAY_STEP == 0:
        print("Epoch:", '%04d,' % (epoch + 1),
              "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

    # get accuracy for validation data
    validation_accuracy = 0
    if i % VALIDATION_STEP == 0:
        # calculate accuracy
        validation_accuracy = sess.run(accuracy,
                                       feed_dict={x: validation_data, y_: validation_labels,
                                                  is_training: False})
        print("Epoch:", '%04d,' % (epoch + 1),
              "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

    return validation_accuracy


if __name__ == '__main__':
    train()
