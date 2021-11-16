import configparser
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import smiley.cnn_model as cnn_model
import smiley.utils as utils

EPOCHS = 0

class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global EPOCHS

        utils.update_progress(100 * epoch / EPOCHS)

def train():
    global EPOCHS

    print("\nCNN TRAINING STARTED.")
    utils.update_progress(1)
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

    MODEL_PATH = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['MODELS'],
                              config['DEFAULT']['IMAGE_SIZE'], config['CNN']['MODEL_FILENAME'])
    IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])
    BATCH_SIZE = int(config['DEFAULT']['TRAIN_BATCH_SIZE'])
    EPOCHS = int(config['CNN']['EPOCHS'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        utils.CATEGORIES_LOCATION,
        color_mode='grayscale',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    model = cnn_model.createModel(train_generator.num_classes)

    model.fit(train_generator, epochs=EPOCHS, callbacks=[ProgressCallback()])
    model.save(MODEL_PATH)
    print("CNN TRAINING END.")