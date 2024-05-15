# Smiley Recognition using TensorFlow

###### A Python app for smiley recognition, in which the training and the classification can be done on the interface using images generated with the mouse, imported from files, or even acquired from a webcam.

![alt text](./gif.gif "Smiley Recognition")

The code is partially based on [this
repo](https://github.com/sugyan/tensorflow-mnist) and developed by Henrique
Orefice and Alexander Abstreiter.

## General knowledge

We provide an easily understandable and concise backend and frontend capable of
generating and importing data (i.e. images), as well as training and testing
Machine Learning models for smiley recognition.

### Requirements

- Python >=3.5 (recommended current)

### Installation
You might want to use [virtual
environments](https://docs.python.org/3/library/venv.html) or `conda` to install
the dependencies.

This could be done e.g. by
```bash
python -m venv ~/.venv/smiley
source ~/.venv/smiley/bin/activate
```

```bash
> # Install the required packages
> pip install -r requirements.txt
>
> # Run the app
> python main.py
>
> # Navigate to localhost:5000
```

### Parameters

Changing important parameters, e.g. learning rates and number of epochs, can be
done on the webapp. Further parameters can be changed in `smiley/config.ini`.
Description of all parameters of `smiley/config.ini`:

##### DEFAULT

- **max_number_user_categories:** number of maximum categories a user can add to
  the application
- **train_batch_size:** number of samples in a single batch
- **train_ratio:** the ratio of how much data will be used for training and how
  much for validation
- **image_size:** the width and height of the network-input images
- **predefined_categories:** the categories which are always available for the
  user

##### CNN

- **learning_rate:** hyper-parameter that controls how much the weights of our
  network are adjusted for an optimizer step
- **epochs:** number of times the entire (train-)dataset is passed forward and
  backward through the network
- **model_filename:** the filename of the stored CNN model

##### DIRECTORIES

- **logic:** contains the important scripts, image data and stored models of the
  program
- **categories:** contains the image data
- **models:** contains the stored models of the program

### License

[MIT License](LICENSE)
