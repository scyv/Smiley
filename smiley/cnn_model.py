import configparser
import os
import tensorflow as tf


def createModel(nCategories):
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    learning_rate = float(config['CNN']['LEARNING_RATE'])
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=(image_size, image_size, 1)))

    addPreprocessing(model)
    addDenseModel(model)
    #addCNNModel(model)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(nCategories))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()

    return model

def addPreprocessing(model):
    model.add(tf.keras.layers.Rescaling(1.0 / 255))
    model.add(tf.keras.layers.RandomFlip(mode='horizontal'))
    model.add(tf.keras.layers.RandomZoom(height_factor=0.2))

def addDenseModel(model):
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))

def addCNNModel(model):
    model.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu'))
