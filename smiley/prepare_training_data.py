import configparser
import os
import math
import numpy
import cv2
import utils
from scipy import ndimage

# parameters
NUM_LABELS = len(utils.update_categories())

# load config params
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))


# get images from category folders, add them to training/test images
def add_data(train_images, train_labels, test_images, test_labels, train_ratio):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])

    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(
        utils.CATEGORIES_LOCATION,
        color_mode='grayscale',
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='binary')

    number_of_images = generator.samples
    number_of_categories = generator.num_classes
    number_processed = 0
    images = []
    labels = []

    # is there any data?
    if number_of_images == 0:
        return None

    # stores how many images of each category are present
    number_per_category = {c: 0.0 for c in range(number_of_categories)}

    while number_processed < number_of_images:
        item = next(generator)
        image = numpy.array(item[0], dtype=numpy.uint8).reshape(1, image_size, image_size, 1)
        #image = (((255 - image) / 255.0) - 0.5)
        #image = numpy.reshape(image, [1, -1])
        label = int(item[1][0])
        number_per_category[label] += 1.0
        labels.append(label)
        images.append(numpy.reshape(image, image_size * image_size))
        number_processed += 1

    # Code with test set
    # stores how many images of each category are in the training set
    #number_per_category_in_training = {c: 0.0 for c in range(NUM_LABELS)}

    for i, x in enumerate(images):
        # Code without test set
        train_images.append(x)
        train_labels.append(labels[i])

        # Code with test set
        #if number_per_category_in_training[category] < number_per_category[category] * train_ratio:
        #    number_per_category_in_training[category] += 1.0
        #    train_images.append(x)
        #    train_labels.append(labels[i])
        #else:
        #    test_images.append(x)
        #    test_labels.append(labels[i])

    train_images = numpy.array(train_images)

    # Code with test set
    #test_images = numpy.array(test_images)

    # transform labels into one-hot vectors
    one_hot_encoding = numpy.zeros((len(train_images), number_of_categories))
    one_hot_encoding[numpy.arange(len(train_images)), train_labels] = 1
    train_labels = numpy.reshape(one_hot_encoding, [-1, number_of_categories])

    # Code with test set
    #one_hot_encoding = numpy.zeros((len(test_images), number_of_categories))
    #one_hot_encoding[numpy.arange(len(test_images)), test_labels] = 1
    #test_labels = numpy.reshape(one_hot_encoding, [-1, number_of_categories])

    # Code without test set
    return train_images, train_labels, None, None

    # Code with test set
    #if sum([1 for c in number_per_category.items() if
    #        c[0] not in [str(n) for n in range(10)] and
    #        c[1] != 0.0 and c[1] == number_per_category_in_training[c[0]]]) == 0:
    #    return train_images, train_labels, test_images, test_labels
    #else:
    #    # at least one category has all examples in the training set (meaning there are not
    #    # enough examples for a training set and a testing set)
    #    for i in range(0, len(number_per_category)):
    #        if number_per_category[i] == number_per_category_in_training[i]:
    #            idx = i
    #    raise Exception("Error while preparing data. Category '" + utils.get_category_names()[idx]
    #                    + "' has just %d images but needs at least %d images." % (int(number_per_category[idx]), 5))


# create a validation set from part of the training data
def create_validation_set(train_data, train_labels, train_ratio):
    train_data_result = []
    train_labels_result = []
    validation_data_result = []
    validation_labels_result = []

    number_per_category = {c: 0.0 for c in range(NUM_LABELS)}
    for i, x in enumerate(train_data):
        category = [z for z in range(len(train_labels[i])) if train_labels[i][z] == 1.0][0]
        number_per_category[category] += 1.0

    number_per_category_in_validation = {c: 0.0 for c in range(NUM_LABELS)}
    number_per_category_in_training = {c: 0.0 for c in range(NUM_LABELS)}

    for i, x in enumerate(train_data):
        category = [z for z in range(len(train_labels[i])) if train_labels[i][z] == 1.0][0]
        if number_per_category_in_training[category] < number_per_category[category] * train_ratio:
            number_per_category_in_training[category] += 1.0
            train_data_result.append(x)
            train_labels_result.append(train_labels[i])
        else:
            number_per_category_in_validation[category] += 1.0
            validation_data_result.append(x)
            validation_labels_result.append(train_labels[i])

    if not number_per_category_in_validation.values():
        raise Exception("Please add at least one category.")
    elif min(number_per_category_in_validation.values()) == 0:
        # at least one of the categories has no items in the validation set (not enough training examples)
        msg = "<b>Error</b> while preparing data:"
        for idx in range(0, len(number_per_category_in_validation.values())):
            if list(number_per_category_in_validation.values())[idx] == 0:
                img = "images" if number_per_category[idx] > 1 else "image"
                msg += " category '<b>" + utils.get_category_names()[idx] + "</b>' has just <b>" + str(int(
                    number_per_category[idx])) + "</b> " + img + ","
        exception_msg = msg + " but at least <b>%d</b> images are required for each category." \
                        % utils.get_number_of_images_required()
        print(exception_msg)
        raise Exception(exception_msg)
    else:
        return numpy.array(train_data_result), numpy.array(train_labels_result), \
               numpy.array(validation_data_result), numpy.array(validation_labels_result)


# augment training data
def expand_training_data(images, labels):
    expanded_images = []
    expanded_labels = []
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    j = 0
    for x, y in zip(images, labels):
        j += 1
        if j % int(config['LOGS']['EXPAND_DISPLAY_STEP']) == 0:
            print('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        image = numpy.reshape(x, (-1, int(config['DEFAULT']['IMAGE_SIZE'])))
        num_augm_per_img = int(config['DEFAULT']['NUMBER_AUGMENTATIONS_PER_IMAGE'])
        max_angle = int(config['DEFAULT']['MAX_ANGLE_FOR_AUGMENTATION'])
        for i in range(num_augm_per_img):
            # shift the image with random distance
            max_shift = int(math.floor(image_size * 0.15))
            shift = numpy.random.randint(-max_shift, max_shift, 2)
            new_img = ndimage.shift(image, shift, cval=255)

            # rotate the image with random degree
            angle = numpy.random.randint(-max_angle, max_angle, 1)
            rotate_matrix = cv2.getRotationMatrix2D(center=(image_size, image_size), angle=int(angle), scale=1)
            new_img = cv2.warpAffine(src=new_img, M=rotate_matrix, dsize=(image_size, image_size), borderValue=(255))

            # zoom image while keeping its dimensions
            zoom = numpy.random.uniform(0.5, 1.5)
            new_img = cv2_clipped_zoom(new_img, zoom)

            path = os.path.join(os.path.dirname(__file__), "data", "exp" + str(j))
            if not os.path.exists(path):
                os.makedirs(path)
            utils.save_image(new_img, path)

            # register new training data
            expanded_images.append(numpy.reshape(new_img, image_size * image_size))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data


# Source: https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = numpy.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(numpy.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)
    const_fill_value = 255

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = numpy.pad(result, pad_spec, mode='constant', constant_values=const_fill_value)
    assert result.shape[0] == height and result.shape[1] == width
    return result


# prepare training data (generated images)
def prepare_data(use_data_augmentation=True):
    global NUM_LABELS, config
    NUM_LABELS = len(utils.update_categories())
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    train_ratio = float(config['DEFAULT']['TRAIN_RATIO'])

    # add data from category folders
    train_data, train_labels, test_data, test_labels = add_data(train_data, train_labels, test_data, test_labels,
                                                                train_ratio)

    # create a validation set
    train_data, train_labels, _, _ = create_validation_set(train_data, train_labels, train_ratio)

    # concatenate train_data and train_labels for random shuffle
    if use_data_augmentation:
        # augment training data by random rotations etc.
        train_total_data = expand_training_data(train_data, train_labels)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)
        numpy.random.shuffle(train_total_data)

    train_size = train_total_data.shape[0]  # size of training set

    return NUM_LABELS, train_total_data, train_size, test_data, test_labels
