import configparser
import math
import os
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, render_template, request
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.framework.errors_impl import NotFoundError

import smiley.utils as utils
import smiley.cnn_train as cnn_train

config = configparser.ConfigParser()
config.file = os.path.join(os.path.dirname(__file__), 'smiley/config.ini')
config.read(config.file)

app = Flask(__name__)

MODELS_DIRECTORY = os.path.join(
    config['DIRECTORIES']['LOGIC'],
    config['DIRECTORIES']['MODELS'],
    config['DEFAULT']['IMAGE_SIZE']
)
MODEL_PATH = os.path.join(
    MODELS_DIRECTORY,
    config['CNN']['MODEL_FILENAME']
)
IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])


def load_categories():
    '''
        Prepares the categories that are stored on disk to be used by
        TensorFlow.
    '''
    global num_categories

    if utils.CATEGORIES_IN_USE is None:
        utils.initialize_categories_in_use()
    else:
        utils.update_categories_in_use()
    num_categories = len(utils.CATEGORIES_IN_USE)


def cnn_predict(input):
    '''
        Calculate a normalized list of predictions for the given input image.
        The given input must be a `np.array` of shape `(1, IMAGE_SIZE,
        IMAGE_SIZE, 1)` and the pixel values MUST not be rescaled (the model
        ensures this).

        Params:
        - input: np.array - the image to classify

        Returns: list of the predicted probabilities
    '''
    model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict(input)
    normalized = tf.nn.softmax(prediction).numpy().flatten()
    return normalized


@app.route('/')
def render_main_page():
    maxNumUserCat = config['DEFAULT']['MAX_NUMBER_USER_CATEGORIES']
    batchSize = config['DEFAULT']['TRAIN_BATCH_SIZE']
    cnnRate = config['CNN']['LEARNING_RATE']
    cnnEpochs = config['CNN']['EPOCHS']
    predefined_categories = (
        config['DEFAULT']['PREDEFINED_CATEGORIES'].split(',')
    )

    data = {
        'image_size': IMAGE_SIZE,
        'batchSize': batchSize,
        'cnnRate': cnnRate,
        'cnnEpochs': cnnEpochs,
        'maxNumUserCat': maxNumUserCat,
        'cats_img_number': utils.get_number_of_images_per_category(),
        'categories': list(set().union(
            utils.get_category_names(),
            predefined_categories
        )),
        'user_categories': list(
            set(utils.get_category_names()) - set(predefined_categories)
        )
    }

    return render_template('index.html', data=data)


# Predict category probabilities
@app.route('/api/classify', methods=['POST'])
def classify():
    # input with pixel values between 0 (black) and 255 (white)
    data = np.array(request.json, dtype=np.uint8)

    # pixels don't need to be rescaled (i.e. to be in range [0, 1]), because
    # the
    # model contains a Rescaling layer that is applied during training and
    # inference
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling

    # final shape is (1, IMAGE_SIZE, IMAGE_SIZE, 1) with (BATCH_SIZE, X, Y,
    # VALUE)
    cnn_input = data.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    err = ''  # string with error messages

    cnn_output = []

    # if no categories available or too few images pro category, print error
    # message
    if len(utils.update_categories()) == 0 or utils.are_images_missing():
        err = utils.get_not_enough_images_error()

    try:
        cnn_output = cnn_predict(cnn_input)
        cnn_output = [-1.0 if math.isnan(f) else f for f in cnn_output]
    except (NotFoundError, InvalidArgumentError):
        err = 'No model found. Please train the network.'

    if utils.is_maybe_old() and len(err) == 0:
        err = 'The model may be outdated. Please retrain the network.'

    return jsonify(
        classifiers=['CNN'],
        results=[np.array(cnn_output).tolist()],
        error=err,
        categories=utils.get_category_names_in_use()
    )


# Add training example
@app.route('/api/add-training-example', methods=['POST'])
def add_training_example():
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    image = np.array(
        request.json['img'],
        dtype=np.uint8
    ).reshape(image_size, image_size, 1)

    category = request.json['cat']
    utils.add_training_example(image, category)

    if utils.are_images_missing():
        err = utils.get_not_enough_images_error()
        return jsonify(error=err)

    return 'ok'


# Delete a category
@app.route('/api/delete-category', methods=['POST'])
def delete_category():
    category = request.json['cat']
    utils.delete_category(category)

    return 'ok'


# Update config parameters
@app.route('/api/update-config', methods=['POST'])
def update_config():
    config.set('CNN', 'LEARNING_RATE', request.json['cnnRate'])
    config.set('CNN', 'EPOCHS', request.json['cnnEpochs'])
    config.set('DEFAULT', 'train_batch_size', request.json['batchSize'])

    # Write config back to file
    with open(config.file, 'w') as f:
        config.write(f)

    return 'ok'


# Train model
@app.route('/api/train-models', methods=['POST'])
@utils.capture
def train_models():
    if len(utils.CATEGORIES) == 0 or utils.are_images_missing():
        err = utils.get_not_enough_images_error()
        return jsonify(error=err)

    utils.update_progress(1)

    utils.set_maybe_old(True)
    load_categories()

    try:
        cnn_train.train()
    except BaseException as trainError:
        print(trainError)
        err = 'Unknown error.'
        return jsonify(error=err)

    if utils.train_should_stop():
        utils.train_should_stop(False)
    else:
        utils.set_maybe_old(False)

    utils.reset_progress()

    return 'ok'


# Retrieve training progress
@app.route('/api/train-progress')
def train_progress():
    progress = utils.get_progress()

    return jsonify(progress=progress)


# Stop the training and delete all saved models
@app.route('/api/stop-training', methods=['POST'])
def stop_training():
    utils.train_should_stop(True)

    return 'ok'


@app.route('/api/get-console-output')
def console_output():
    output = utils.LOGGER.pop()

    return jsonify(out=output)


@app.route('/api/open-category-folder', methods=['POST'])
def open_category_folder():
    category = request.json['cat']
    try:
        utils.open_category_folder(category)
    except AttributeError as e:
        print(e)
        err = (
            "Could not open the folder in the system file explorer. " +
            "This function is only supported on Windows."
        )
        return jsonify(error=err), 500

    return 'ok'


# main
if __name__ == '__main__':
    # Open webbrowser tab for the app
    # webbrowser.open_new_tab('http://localhost:5000')
    if not os.path.exists(MODELS_DIRECTORY):
        os.makedirs(MODELS_DIRECTORY)
    load_categories()
    app.run(host='0.0.0.0')
