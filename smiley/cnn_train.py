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
    EPOCHS = int(config['CNN']['EPOCHS'])

    # get training/validation/testing data
    try:
        curr_number_of_categories, train_total_data, train_size, test_data, test_labels = prepare_training_data.prepare_data(True)
    except Exception as inst:
        raise Exception(inst.args[0])

    train_images = numpy.reshape(train_total_data[:, :-curr_number_of_categories], (train_size, -1, IMAGE_SIZE))
    train_labels = train_total_data[:, -curr_number_of_categories:]
    model = cnn_model.createModel(curr_number_of_categories)
    
    model.fit(train_images, train_labels.T[1], epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(MODEL_PATH)
    model.summary()
    print("CNN TRAINING END.")