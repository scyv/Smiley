import configparser
import os
import tensorflow as tf


# Convolutional Neural Network with two convolutional and max pool layers,
# followed by two fully connected layers with dropout on the first one.
# At the end, softmax is applied to transform the values into probabilities for each class.
def createModel(nCategories):
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    learning_rate = float(config['CNN']['LEARNING_RATE'])
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(nCategories))

    #model.add(tf.keras.layers.Flatten(input_shape=(image_size, image_size)))
    #model.add(tf.keras.layers.Dense(128, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(nCategories))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model
