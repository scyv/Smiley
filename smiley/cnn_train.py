import configparser
import os
import tensorflow as tf

import smiley.cnn_model as cnn_model
import smiley.utils as utils

EPOCHS = 0


class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global EPOCHS

        utils.update_progress(100 * epoch / EPOCHS)


def train():
    '''
        Creates a new model, fits it to the collected training data and stores
        it in the location specified in the config.ini.

        The models network topology depends on `cnn_model.py`. Refer to this
        file to see how the model is created and which features can be enabled
        or disabled.

        The model's hyper parameters are fetched from the config file.
        `train()` assumes a directory of images per category (label) in the
        configured categories location. All images are interpreted as grayscale
        images and the labels are stored in integer encoding.

        If a model already exists at the location specified in config.ini, it
        is overwritten.

        Returns: void
    '''
    global EPOCHS

    print('\nTRAINING STARTED.')
    utils.update_progress(1)
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        config['DIRECTORIES']['MODELS'],
        config['DEFAULT']['IMAGE_SIZE'],
        config['CNN']['MODEL_FILENAME']
    )
    IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])
    BATCH_SIZE = int(config['DEFAULT']['TRAIN_BATCH_SIZE'])
    EPOCHS = int(config['CNN']['EPOCHS'])

    data = tf.keras.utils.image_dataset_from_directory(
        directory=utils.CATEGORIES_LOCATION,
        label_mode='int',
        color_mode='grayscale',
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    model = cnn_model.createModel(len(data.class_names))

    model.fit(data, epochs=EPOCHS, callbacks=[ProgressCallback()])
    model.save(MODEL_PATH)
    print('TRAINING END.')
