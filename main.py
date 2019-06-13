import configparser
import math
import os
import sys
import webbrowser
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError

sys.path.append('smiley')
import regression_model, cnn_model, regression_train, cnn_train, utils

config = configparser.ConfigParser()
config.file = os.path.join(os.path.dirname(__file__), 'smiley/config.ini')
config.read(config.file)

MODELS_DIRECTORY = os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['MODELS'],
                                config['DEFAULT']['IMAGE_SIZE'])
IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])

# create folder for models if it doesn't exist
if not os.path.exists(MODELS_DIRECTORY):
    os.makedirs(MODELS_DIRECTORY)


# updates the models if the number of classes changed
def maybe_update_models():
    global y1, variables, saver_regression, y2, saver_cnn, x, is_training, sess, num_categories

    # close (old) tensorflow if existent
    if 'sess' in globals():
        sess.close()

    if utils.CATEGORIES_IN_USE is None:
        utils.initialize_categories_in_use()
    else:
        utils.update_categories_in_use()
    num_categories = len(utils.CATEGORIES_IN_USE)

    # Model variables
    x = tf.placeholder("float", [None, IMAGE_SIZE * IMAGE_SIZE])  # image input placeholder
    is_training = tf.placeholder("bool")  # used for activating the dropout in training phase

    # Tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Regression model
    y1, variables = regression_model.regression(x, nCategories=num_categories)  # prediction results and variables
    saver_regression = tf.train.Saver(variables)

    # CNN model
    y2, variables = cnn_model.convolutional(x, nCategories=num_categories, is_training=is_training)  # prediction results and variables
    saver_cnn = tf.train.Saver(variables)


# Initialize the categories mapping, the tensorflow session and the models
maybe_update_models()


# Regression prediction
def regression_predict(input):
    saver_regression.restore(sess, os.path.join(MODELS_DIRECTORY, config['REGRESSION']['MODEL_FILENAME']))
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


# CNN prediction
def cnn_predict(input):
    saver_cnn.restore(sess, os.path.join(MODELS_DIRECTORY, config['CNN']['MODEL_FILENAME']))
    return sess.run(y2, feed_dict={x: input, is_training: False}).flatten().tolist()


# Webapp definition
app = Flask(__name__)


# Root
@app.route('/')
def main():
    maxNumUserCat = config['DEFAULT']['MAX_NUMBER_USER_CATEGORIES']
    numAugm = config['DEFAULT']['NUMBER_AUGMENTATIONS_PER_IMAGE']
    batchSize = config['DEFAULT']['TRAIN_BATCH_SIZE']
    srRate = config['REGRESSION']['LEARNING_RATE']
    srEpochs = config['REGRESSION']['EPOCHS']
    cnnRate = config['CNN']['LEARNING_RATE']
    cnnEpochs = config['CNN']['EPOCHS']
    predefined_categories = config['DEFAULT']['PREDEFINED_CATEGORIES'].split(",")
    
    data = {'image_size': IMAGE_SIZE, 'numAugm': numAugm, 'batchSize': batchSize, 'srRate': srRate,
            'srEpochs': srEpochs, 'cnnRate': cnnRate, 'cnnEpochs': cnnEpochs, 'maxNumUserCat': maxNumUserCat,
            'cats_img_number': utils.get_number_of_images_per_category(),
            'categories': list(set().union(utils.get_category_names(), predefined_categories)),
            'user_categories': list(set(utils.get_category_names()) - set(predefined_categories))}
    
    return render_template('index.html', data=data)


# Predict category probabilities
@app.route('/api/classify', methods=['POST'])
def classify():
    # input with pixel values between 0 (black) and 255 (white)
    data = np.array(request.json, dtype=np.uint8)

    # transform pixels to values between 0 (white) and 1 (black)
    regression_input = ((255 - data) / 255.0).reshape(1, IMAGE_SIZE * IMAGE_SIZE)

    # transform pixels to values between -0.5 (white) and 0.5 (black)
    cnn_input = (((255 - data) / 255.0) - 0.5).reshape(1, IMAGE_SIZE * IMAGE_SIZE)

    err = ""  # string with error messages

    regression_output = []
    cnn_output = []

    # if no categories available or too few images pro category, print error message
    if len(utils.update_categories()) == 0 or utils.not_enough_images():
        err = utils.get_not_enough_images_error()

    try:
        regression_output = regression_predict(regression_input)
        regression_output = [-1.0 if math.isnan(b) else b for b in regression_output]

        cnn_output = cnn_predict(cnn_input)
        cnn_output = [-1.0 if math.isnan(f) else f for f in cnn_output]
    except (NotFoundError, InvalidArgumentError):
        err = "No model found. Please train the network."

    if utils.is_maybe_old() and len(err) == 0:
        err = "The model may be outdated. Please retrain the network for updated results."

    return jsonify(classifiers=["Regression", "CNN"], results=[regression_output, cnn_output],
                   error=err, categories=utils.get_category_names_in_use())


# Add training example
@app.route('/api/add-training-example', methods=['POST'])
def add_training_example():
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    image = np.array(request.json["img"], dtype=np.uint8).reshape(image_size, image_size, 1)
    category = request.json["cat"]
    utils.add_training_example(image, category)

    if utils.not_enough_images():
        err = utils.get_not_enough_images_error()
        return jsonify(error=err)
    
    return "ok"


# Delete a category
@app.route('/api/delete-category', methods=['POST'])
def delete_category():
    category = request.json["cat"]
    utils.delete_category(category)

    return "ok"


# Update config parameters
@app.route('/api/update-config', methods=['POST'])
def update_config():
    config.set("CNN", "LEARNING_RATE", request.json["cnnRate"])
    config.set("REGRESSION", "LEARNING_RATE", request.json["srRate"])
    config.set("CNN", "EPOCHS", request.json["cnnEpochs"])
    config.set("REGRESSION", "EPOCHS", request.json["srEpochs"])
    config.set("DEFAULT", "number_augmentations_per_image", request.json["numAugm"])
    config.set("DEFAULT", "train_batch_size", request.json["batchSize"])

    # Write config back to file
    with open(config.file, "w") as f:
        config.write(f)

    return "ok"


# Train model
@app.route('/api/train-models', methods=['POST'])
@utils.capture
def train_models():
    if len(utils.CATEGORIES) == 0 or utils.not_enough_images():
        err = utils.get_not_enough_images_error()
        return jsonify(error=err)

    utils.delete_all_models()
    utils.set_maybe_old(True)
    maybe_update_models()

    try:
        regression_train.train()
        cnn_train.train()
    except:
        err = "Unknown error."
        utils.delete_all_models()
        return jsonify(error=err)

    if utils.train_should_stop():
        utils.delete_all_models()
        utils.train_should_stop(False)
    else:
        utils.set_maybe_old(False)

    utils.reset_progress()

    return "ok"


# Retrieve training progress
@app.route('/api/train-progress')
def train_progress():
    progress = utils.get_progress()

    return jsonify(progress=progress)


# Stop the training and delete all saved models
@app.route('/api/stop-training', methods=['POST'])
def stop_training():
    utils.train_should_stop(True)

    return "ok"


@app.route('/api/get-console-output')
def console_output():
    output = utils.LOGGER.pop()
    
    return jsonify(out=output)


@app.route('/api/open-category-folder', methods=['POST'])
def open_category_folder():
    category = request.json["cat"]
    utils.open_category_folder(category)

    return "ok"


# main
if __name__ == '__main__':
    # Open webbrowser tab for the app
    webbrowser.open_new_tab("http://localhost:5000")

    app.run(host="0.0.0.0")
