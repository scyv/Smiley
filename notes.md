# Notes
## Changes
### Updates to dependencies
I've updated all dependencies to match the most recent stable versions.

### `ImageDataGenerator` is deprecated
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
I've replaced this with `tf.keras.utils.image_dataset_from_directory()`. Also,
the ImageDataGenerator previously contained the rescaling (mapping to range
0-1), random shearing, zooming and flipping. This is now part of preprocessing
steps of the model itself (see `addPreprocessing()` for this). This could be
commented out to outline the importance of preprocessing. Unfortunately, the
random shear is currently not implemented as I wasn't able to get the
corresponding `keras_cv` function to work.

### Scaling/Shaping changes for prediction
I've found that the actual prediction rescaled the pixel values to a range
-0.5-0.5. I don't understand why this is necessary; I even believe that this is
wrong. The model was previously trained on 0-1 range, why predict with a
differen range? Anyway, by adding the Rescale-Layer in the preprocessing step,
the rescaling is automatically performed, also for inference (see
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling). Thus,
there is no need to scale data that is fed into the model for prediction. 

While doing that, I also changed the reshaping such that we have one single
reshaping of the image data that produces the `(batch_size, width, height, b/w)`
shape needed for prediction instead of two calls to reshape at different places
in the code.

## Misc
- Maybe the `/tmp` might be too small to install Tensorflow directly. Workaround
  with `export TMPDIR=/path/to/manual/tmp/dir`
- "AttributeError: module 'os' has no attribute 'startfile" when downloading
  training data (seems to be a Linux issue)
-  not all images can be loaded (issue with `neutral/._R7bHOIpAxC.png`)