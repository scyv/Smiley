import configparser
import os
import tensorflow as tf


def createModel(nCategories):
    '''
        Creates a model for the given number of categories. Parameters of the
        model are taken from "config.ini", e.g. the learning rate or the image
        size.

        The model consists of four tiers of layers:
        1. The input layer
        2. The (optional) preprocessing layers
        3. Either dense layers OR convolutional layers
        4. The output layers

        The layers can be switched by (un)commenting the respective lines.

        Params:
        - nCategories: int - the number of categories the final model should
        be able to predict.

        Returns:
        a tf.keras.Sequential model
    '''
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    learning_rate = float(config['CNN']['LEARNING_RATE'])
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=(image_size, image_size, 1)))

    #addPreprocessing(model)
    addDenseModel(model)
    #addCNNModel(model)

    model.add(tf.keras.layers.Flatten())
   # model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(nCategories))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    return model


def addPreprocessing(model):
    '''
        Creates a tier of preprocessing layers and adds them to the given
        model. The preprocessing consists of

        1. A rescaling layer to map inputs to [0,1] (applied during training
           and inference)
        2. A random flip layer that flips images horizontally at random
           (applied during training)
        3. A random zoom layer that zooms into the images at random (applied
           during training)

        Params: - model: tf.keras.Model - the model to which the tier should be
        added

        Returns: void
    '''
    model.add(tf.keras.layers.Rescaling(1.0 / 255))
    model.add(tf.keras.layers.RandomFlip(mode='horizontal'))
    model.add(tf.keras.layers.RandomZoom(height_factor=0.2))


def addDenseModel(model):
    '''
        Adds three dense layers with ReLU activation to the given model.

        Params:
        - model: tf.keras.Model - the model to which the tier should be added

        Returns:
        void
    '''
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))


def addCNNModel(model):
    '''
        Adds three convolution layers and two max pooling layers between them
        to the given model.

        Params:
        - model: tf.keras.Model - the model to which the tier should be
        added

        Returns: void
    '''
    model.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu'))
