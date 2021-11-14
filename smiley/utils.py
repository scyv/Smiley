import configparser
import os
import sys
import png
import math
import shutil
import string
import random

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

MODELS_DIRECTORY = os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['MODELS'],
                                config['DEFAULT']['IMAGE_SIZE'])
CATEGORIES_LOCATION = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['CATEGORIES'],
                                       config['DEFAULT']['IMAGE_SIZE'] + "/")
#In config: [DIRECTORIES] categories = T:\categories
#CATEGORIES_LOCATION = os.path.join(config['DIRECTORIES']['CATEGORIES'],
#                                       config['DEFAULT']['IMAGE_SIZE'] + "/")

CATEGORIES = None
CATEGORIES_IN_USE = None
MAYBE_OLD_VERSION = False
PROGRESS = {
    'value': 100,
    'process_dist': [0.1,0.9],
    'current_process': 0,
    'previous_value': 0,
    'stop': False
}


def is_maybe_old():
    global MAYBE_OLD_VERSION

    return MAYBE_OLD_VERSION


def set_maybe_old(value):
    global MAYBE_OLD_VERSION

    MAYBE_OLD_VERSION = value


def get_progress():
    global PROGRESS

    return PROGRESS['value']


def update_progress(value):
    global PROGRESS

    PROGRESS['value'] = PROGRESS['previous_value'] + (100*value*PROGRESS['process_dist'][PROGRESS['current_process']])
    
    # if process is completed, add its contribution to previous_value
    if value == 1:
        PROGRESS['previous_value'] += 100*PROGRESS['process_dist'][PROGRESS['current_process']]
        PROGRESS['current_process'] += 1

    return PROGRESS['value']


def train_should_stop(stop='MISSING'):
    global PROGRESS

    if stop != 'MISSING':
        PROGRESS['stop'] = stop

    return PROGRESS['stop']


def reset_progress():
    global PROGRESS

    PROGRESS['value'] = 100
    PROGRESS['current_process'] = 0
    PROGRESS['previous_value'] = 0


# Class for log handling
class Logger(object):
    def __init__(self):
        self.buffer = ""

    def start(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def end(self):
        sys.stdout = self.stdout

    def write(self, data):
        self.buffer += data
        self.stdout.write(data)

    def flush(self):
        pass

    def pop(self):
        length = len(self.buffer)
        out = self.buffer[:length]
        self.buffer = self.buffer[length:]
        return out


# logger object
LOGGER = Logger()


# Decorator to capture standard output
def capture(f):
    def captured(*args, **kwargs):
        LOGGER.start()
        try:
            result = f(*args, **kwargs)
        finally:
            LOGGER.end()
        return result  # captured result from decorated function
    return captured


def delete_all_models():
    for f in os.listdir(MODELS_DIRECTORY):
        os.remove(os.path.join(MODELS_DIRECTORY, f))


def update_categories():
    global CATEGORIES

    # create folder for categories if it doesn't exist:
    if not os.path.exists(CATEGORIES_LOCATION):
        os.makedirs(CATEGORIES_LOCATION)

    # dictionary with keys = category names (from folders), values = category indices
    CATEGORIES = {x: i for (i, x) in
                  enumerate(sorted([z[0].split("/")[-1] for z in os.walk(CATEGORIES_LOCATION) if
                                    z[0].count("/") == CATEGORIES_LOCATION.count("/") and len(
                                        z[0].split("/")[-1]) > 0]))}
    return CATEGORIES


def update_categories_in_use():
    global CATEGORIES_IN_USE

    CATEGORIES_IN_USE = CATEGORIES


def add_training_example(image, category):
    # create folder for category if it doesn't exist:
    path = os.path.join(CATEGORIES_LOCATION, category)
    if not os.path.exists(path):
        os.makedirs(path)

    save_image(image, path)
    update_categories()
    set_maybe_old(True)


def save_image(image, path):
    # name for new training example image
    randomId = ''.join([random.choice(string.ascii_letters 
            + string.digits) for n in range(10)])

    # store new training example image
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    w = png.Writer(image_size, image_size, greyscale=True)
    w.write(open(path + "/" + randomId + ".png", "wb"), image)


# Deletes the folder of the category
def delete_category(category):
    path = os.path.join(CATEGORIES_LOCATION, category)
    shutil.rmtree(path)
    update_categories()
    set_maybe_old(True)


def get_category_names():
    return list(CATEGORIES.keys())


def get_category_names_in_use():
    return list(CATEGORIES_IN_USE.keys())


# Initialize categories in use so that it contains all categories which have enough images
def initialize_categories_in_use():
    global CATEGORIES_IN_USE

    CATEGORIES_IN_USE = {c: i for (i, c) in enumerate(c for (c, n) in get_number_of_images_per_category().items() if n >= get_number_of_images_required())}


# Returns dictionary with keys = category names (from folders), values = number of images for the category
def get_number_of_images_per_category():
    update_categories()
    cat_images = {}
    for z in os.walk(CATEGORIES_LOCATION):
        if len(str(z[0].split("/")[-1])) > 0:
            for x in os.walk(os.path.join(CATEGORIES_LOCATION, str(z[0].split("/")[-1]))):
                # the number of png-files in the category folder is used for the number of images for that category
                n_pngs = len([a for a in x[-1] if a.split(".")[-1] == "png"])
                if n_pngs == 0:
                    delete_category(str(x[0].split("/")[-1]))
                else:
                    cat_images[str(x[0].split("/")[-1])] = n_pngs
    return cat_images


# Returns the number of images required for each category
def get_number_of_images_required():
    # calculating number of images required for each category (-0.000001 for float precision errors)
    # (for with test set add 1)
    return math.ceil((1.0 / (1.0 - float(config['DEFAULT']['train_ratio']))) - 0.000001)


# Returns a string error message that a category has to be added
def get_not_enough_images_error():
    req_images_per_cat = get_number_of_images_required()
    return "Please add at least <b>%d</b> images to each non-empty category and (re-)train the network for updated results." % req_images_per_cat


# Checks if at least one category has not the least required number of images
def not_enough_images():
    req_images_per_cat = get_number_of_images_required()
    cat_img = get_number_of_images_per_category()
    return any(cat_img[cat] < req_images_per_cat for cat in cat_img.keys())


def open_category_folder(category):
    dir = os.path.abspath(os.path.join(CATEGORIES_LOCATION, category))
    os.startfile(dir)
