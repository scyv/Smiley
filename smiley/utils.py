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

MODELS_DIRECTORY = os.path.join(
    config['DIRECTORIES']['LOGIC'],
    config['DIRECTORIES']['MODELS'],
    config['DEFAULT']['IMAGE_SIZE']
)
CATEGORIES_LOCATION = os.path.join(
    os.path.dirname(__file__),
    config['DIRECTORIES']['CATEGORIES'],
    config['DEFAULT']['IMAGE_SIZE'] + '/'
)

CATEGORIES = None
CATEGORIES_IN_USE = None
MAYBE_OLD_VERSION = False
PROGRESS = {
    'value': 100,
    'previous_value': 0,
    'stop': False
}


def is_maybe_old():
    '''
        Returns true if the trained model might be outdated. The model might be
        outdated if the number of categories changed, the training data was
        modified or another process gave some indication for that.
    '''
    global MAYBE_OLD_VERSION

    return MAYBE_OLD_VERSION


def set_maybe_old(value):
    '''
        Stores the given flag to indicate if the trained model might be
        outdated.

        Params:
        - value: bool - the value to set the flag to
    '''
    global MAYBE_OLD_VERSION

    MAYBE_OLD_VERSION = value


def get_progress():
    '''
        Returns the current progress value.
    '''
    global PROGRESS

    return PROGRESS['value']


def update_progress(value):
    '''
        Updates the progress bar to the given value.

        Params:
        - value: int - the new progress value
    '''
    global PROGRESS

    PROGRESS['value'] = value
    return PROGRESS['value']


def train_should_stop(stop='MISSING'):
    '''
        Set the flag to indicate if the current training process should abort
        at the next possible moment.

        Params:
        - stop: bool - the flag to indicate if the training process
        should stop
    '''
    global PROGRESS

    if stop != 'MISSING':
        PROGRESS['stop'] = stop

    return PROGRESS['stop']


def reset_progress():
    '''
        Resets the global `PROGRESS` variable to its initial state,
        i.e. to
        - `value = 100`
        - `current_process = 0`
        - `previous_value = 0`
    '''
    global PROGRESS

    PROGRESS['value'] = 100
    PROGRESS['current_process'] = 0
    PROGRESS['previous_value'] = 0


# Class for log handling
class Logger(object):
    def __init__(self):
        self.buffer = ''

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
    '''
        Deletes all stored models from `MODEL_DIRECTORY` by removing all files
        that are stored in that directory.
    '''
    for f in os.listdir(MODELS_DIRECTORY):
        os.remove(os.path.join(MODELS_DIRECTORY, f))


def update_categories():
    '''
        Initializes the `CATEGORIES` variable such that it is a dictionary of
        `{category: index}`. For this, all directories in the
        `CATEGORIES_LOCATION` are treated as a category.
    '''
    global CATEGORIES

    # create folder for categories if it doesn't exist:
    if not os.path.exists(CATEGORIES_LOCATION):
        os.makedirs(CATEGORIES_LOCATION)

    # dictionary that maps {category: index}
    CATEGORIES = {
        x: i for (i, x) in enumerate(
            sorted([
                entry.name for entry in os.scandir(CATEGORIES_LOCATION)
                if entry.is_dir()
            ])
        )
    }
    return CATEGORIES


def update_categories_in_use():
    '''
        Updates the `CATEGORIES_IN_USE` variable such that it equals the
        `CATEGORIES` variable.
    '''
    global CATEGORIES_IN_USE

    CATEGORIES_IN_USE = CATEGORIES


def add_training_example(image, category):
    '''
        Adds the given image as a training data point to the given category. If
        the category does not yet exist, it is created.

        Params:
        - image: np.Array - image array of shape (size, size, channels)
        - category: str - the name of the category to which the image should
        be added
    '''
    # create folder for category if it doesn't exist:
    path = os.path.join(CATEGORIES_LOCATION, category)
    if not os.path.exists(path):
        os.makedirs(path)

    save_image(image, path)
    update_categories()
    set_maybe_old(True)


def save_image(image, path):
    '''
        Saves the image in the given path on the file system. The file name is
        randomly generated. The image will be stored as a `.png` file.

        Params:
        - image: np.Array - image array of shape (size, size, channels)
        - path: str - the path where to store the image
    '''
    # name for new training example image
    randomId = ''.join([random.choice(
        string.ascii_letters + string.digits
    ) for _ in range(10)])

    # store new training example image
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    w = png.Writer(image_size, image_size, greyscale=True)
    w.write(open(path + '/' + randomId + '.png', 'wb'), image)


# Deletes the folder of the category
def delete_category(category):
    '''
        Deletes the given category by deleting the directory from the file
        system if it exists.

        Params:
        - category: str - the name of the category to delete
    '''
    path = os.path.join(CATEGORIES_LOCATION, category)
    shutil.rmtree(path)
    update_categories()
    set_maybe_old(True)


def get_category_names():
    '''
        Returns a list containing the names of all categories. Not all of the
        returned categories are actually in use. Refer to
        `get_category_names_in_use()` for this.
    '''
    return list(CATEGORIES.keys())


def get_category_names_in_use():
    '''
        Returns a list containing the names of all categories that are in use.
        This is defined by `initialize_categories_in_use()`. Refer to
        `get_category_names()` to obtain the names of all categories.
    '''
    return list(CATEGORIES_IN_USE.keys())


def initialize_categories_in_use():
    '''
        Initializes the CATEGORIES_IN_USE variable such that it is a dictionary
        of `{category: index}` for all categories that reach the minimum number
        of images that are required to train the model.
    '''
    global CATEGORIES_IN_USE

    CATEGORIES_IN_USE = {
        c: i for (i, c) in enumerate(
            c for (c, n) in get_number_of_images_per_category().items()
            if n >= get_number_of_images_required()
        )
    }


def get_number_of_images_per_category():
    '''
        Counts the number of images that are present for each category by
        opening the corresponding folder in the file system and counting all
        files with ".png" suffix.

        Returns a dictionary of `{category_name: number_of_images}`
    '''
    update_categories()
    cat_images = {}
    for category in os.scandir(CATEGORIES_LOCATION):
        if category.is_dir():
            pngs = [
                file.name for file in os.scandir(category)
                if file.is_file() and file.name.endswith('.png')
            ]
            n_pngs = len(pngs)
            if n_pngs == 0:
                delete_category(category.name)
            else:
                cat_images[category.name] = n_pngs

    return cat_images


# Returns the number of images required for each category
def get_number_of_images_required():
    '''
        Calculates the number of images that are required to obtain the
        `train_ratio` that is defined in the config.
    '''
    # -0.000001 for float precision errors
    return math.ceil(
        (1.0 / (1.0 - float(config['DEFAULT']['train_ratio']))) - 0.000001
    )


# Returns a string error message that a category has to be added
def get_not_enough_images_error():
    '''
        Returns an error string indicating that the user needs to add more
        images before the model will yield meaningful results.

        Returns: str
    '''
    req_images_per_cat = get_number_of_images_required()
    return (
        f'Please add at least <b>{req_images_per_cat}</b> ' +
        'images to each non-empty category and (re-)train the ' +
        'network for updated results'
    )


# Checks if at least one category has not the least required number of images
def are_images_missing():
    '''
        Checks if there exists a category that has fewer images than the
        minimum threshold requires.

        Returns True if images are still missing.
    '''
    req_images_per_cat = get_number_of_images_required()
    cat_img = get_number_of_images_per_category()
    return any(cat_img[cat] < req_images_per_cat for cat in cat_img.keys())


def open_category_folder(category):
    '''
        Opens the system file explorer in the directory of the given category.
        Only supported on Windows machines.

        Params:
        - category: str - the name of the category to show
    '''
    dir = os.path.abspath(os.path.join(CATEGORIES_LOCATION, category))
    os.startfile(dir)
