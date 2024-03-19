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
    global EPOCHS

    print('\nCNN TRAINING STARTED.')
    utils.update_progress(1)
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

    MODEL_PATH = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['MODELS'],
                              config['DEFAULT']['IMAGE_SIZE'], config['CNN']['MODEL_FILENAME'])
    IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])
    BATCH_SIZE = int(config['DEFAULT']['TRAIN_BATCH_SIZE'])
    EPOCHS = int(config['CNN']['EPOCHS'])

    data = tf.keras.utils.image_dataset_from_directory(
        directory=utils.CATEGORIES_LOCATION,
        label_mode='int', #needed by SparseCategoricalCrossentropy in the model
        color_mode='grayscale',
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    model = cnn_model.createModel(len(data.class_names))

    model.fit(data, epochs=EPOCHS, callbacks=[ProgressCallback()])
    model.save(MODEL_PATH)
    print('CNN TRAINING END.')