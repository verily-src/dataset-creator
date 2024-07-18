<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'itayr' reviewed: '2023-09-19' }
*-->

# DatasetCreator

Developed and maintained by the nice folks at [Verily](https://verily.com/).

[![Presubmits](https://github.com/verily-src/dataset-creator/actions/workflows/run_presubmit.yaml/badge.svg?branch=main)](https://github.com/verily-src/dataset-creator/actions/workflows/run_presubmit.yaml)
[![Tests](https://github.com/verily-src/dataset-creator/actions/workflows/run_tests.yaml/badge.svg?branch=main)](https://github.com/verily-src/dataset-creator/actions/workflows/run_tests.yaml)
[![Coverage badge](https://github.com/verily-src/dataset-creator/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/verily-src/dataset-creator/tree/python-coverage-comment-action-data)

Installation:
```sh
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "git+ssh://git@github.com/verily-src/dataset-creator.git"
```

## What is DatasetCreator?

`DatasetCreator` is a tool that allows creating PyTorch / Tensorflow datasets in
an easy, reproducible fashion. To get a new dataset, a user only needs to create
a python generator that specifies "recipes" for examples that should be included
in that dataset.

`DatasetCreator` supports 5 main functionalities:

1.  `DatasetCreator.create_example_bank()`: Creates a beam pipeline to populate
    the Examples yielded from the generator, and saves all populated examples to
    an example bank, containing all of the Examples' features.

2.  `DatasetCreator.get_dataset_metadata()`: Aggregates over the entire
    generator. THIS WILL VALIDATE YOUR GENERATOR, and also aggregate over
    examples' metadata. Any metadata key that contains 'label' as a substring
    will be counted.

3.  `DatasetCreator.get_tf_dataset()` / `DatasetCreator.get_torch_dataset()`:
    Pulls examples and frames from an example bank, or creates the dataset
    dynamically. Please note that PyTorch datasets cannot be directly shuffled.
    Dynamic datasets are inherently not compatible with shuffling as they fire
    examples as they are ready. Static datasets shuffling will be supported in
    the future.

4.  `DatasetCreator.load_latest_dataset('my_dataset')`: Reproduces the examples
    of the last registered dataset. Another option to reproduce an older dataset
    is by calling:

5.  `dataset_creator.get_registered_datasets('my_dataset')`: This returns a
    DataFrame containing all creation times of datasets with name
    `'my_dataset'`. The dataset can then be reproduced by:
    `DatasetCreator(creation_time=registered_df['creation_time'][X])`

## Creating a Dataset the first time

First, a user must create a generator yielding `dataset_creator.Example`
objects. An example is intuitively just an `ImmutableOrderedDict`. It is a
mapping from feature names to the features that build the example.

The builtin python types that can be used as features in an `Example` are
`bytes`, `str`, `int`, `float`, `bool`, Sequences of those, and `None`. The only
other supported features are float `np.ndarray`, `tf.Tensor`, and
`CustomFeature`s that are registered in `example_lib._TYPE_NAME_TO_TYPE`.

*IMPORTANT NOTE* The generator should avoid performing I/O as much as possible.
It's better to load I/O resources needed (e.g. video lengths) ahead of
generation time, and use the pre-computed lengths in the generator.

An example of such generator is given below:

```python
import numpy as np
import os
from datetime import datetime

from dataset_creator.video_io import video_io
from dataset_creator.dataset_creator import dataset_creator
from dataset_creator.dataset_creator.features import images_feature


DATASET_NAME = 'test_dataset'
_ROOT_DIR = '/gcs/data-bucket/test_data'


def test_dataset_generator():
  for fn in os.listdir(_ROOT_DIR):
    if not fn.endswith('.mp4'):
      continue
    video_path = os.path.join(_ROOT_DIR, fn)
    # Set check_path to False to avoid I/O so the generator is fast.
    reader = video_io.VideoFileReader(video_path, check_path=False)

    for t in np.arange(0, 60000, 1000):
      yield dataset_creator.Example(
          {
              'label': int(t / 1000) % 10,
              'frames': images_feature.ImagesFeature(
                  reader=reader,
                  read_by=images_feature.READ_BY_TIMESTAMP_MILLIS,
                  read_at=[int(t)],
                  image_size=(224, 224)
              ),
          }
      )
```

Please note that `int(t)` is passed to the Example, as np.int64 is not
primitive. Given that generator, the instantiation of the DatasetCreator is
simply given by:

```python
creator = dataset_creator.DatasetCreator(DATASET_NAME, test_dataset_generator)
```

IMPORTANT NOTE: A dataset is identified by its name and its creation time. If
you create the "same" dataset by instantiating it twice with the same generator
in two different times, the two instances will be treated as different datasets
(and thus might be saved twice). To avoid this, please refer to
`DatasetCreator.load_latest_dataset` and
`dataset_creator.get_registered_datasets`.

### max_examples parameter

The max_examples parameter can be passed to the DatasetCreator instantiation.
This parameter will islice your generator to the number of requested examples.
The default value is dataset_creator.MAX_EXAMPLES.

### Validating your dataset (by aggregating it)

Once you have your creator at hand, simply call:

```python
creator.get_dataset_metadata()
```

This will run over your entire generator (validating it along the way) and count
the labels it encounters. THIS PROCESS MAY TAKE A FEW MINUTES (for my generator,
it took ~6 minutes for looping over ~1,000,000 examples)

You can also pass an optional parameter `bucketizer` to count something other
than the labels. This parameter is a function and will be treated as the
"extractor" of the data you wish accumulate over. For example, if
`example['annotators']` contains a list of all annotators involved in
this example, you can pass:

```python
def bucketizer(example: dataset_creator.Example) -> list[Hashable]:
  return example.metadata.get('annotators', [])
```

This will aggregate and show you which annotator is the most prominent in your
dataset.

## Creating an example bank

If you wish to dynamically create your dataset, feel free to proceed to the next
section.

After you've validated your generator, you can create an example bank containing
your examples and all their features:

```python
creator.create_example_bank()
```

The process of creating an example bank *ALWAYS* starts by registering all the
given examples to a DB. Please note that the example bank creation process
relies on the examples being saved in the DB.

As a basic benchmark number, the process of creating an example bank should take
(end-to-end) ~1.5 hours for a dataset with 1.5M examples.


## Creating a tf.data.Dataset instance

Once you've created your example bank (or you want to dynamically work with your
dataset) you can get a tf.data.Dataset object which contains the frames you've
requested.

The Dataset has the following keys:

1.   For primitive types, the same key as the one you provided in your
     generator. In our case, the key `'label'` will hold a tf.Tensor with the
     corresponding label of the `Example`.
2.   For CustomFeatures, such as `ImagesFeature` or `InferenceFeature`, the
     dataset might hold several keys for each features. The structure of each of
     those keys will be <feature_name>/<sub_key>, where the sub keys can be
     identified by the CustomFeature's output_signature member. For example, in
     our case the keys corresponding to the `'frames'` feature, are
     `'frames/images'`, `'frames/images_timestamp'`, and `'frames/images_path'`.

The get_tf_dataset method will try and find an appropriate example bank to pull
examples from, and if it cannot find one it will fall back to dynamically
creating the dataset. Please note that this fallback might be slower than using
an example bank.

get_static_tf_dataset and get_dynamic_tf_dataset can be called directly as well
to make sure you run in the expected mode.

The retrieved dataset in our example is a Mapping with the following keys:

1. `dataset['label']` - contains the index labels passed in each Example.

3. `dataset['frames/images']` - contains (after mapping) the decoded frames.In
   case a single frame is requested in each batch the shape is (N, H, W, C). In
   case L>1 frames were requested in every sequence the shape will be
   (N, L, H, W, C).

4. `dataset['frames/images_read_at']` - contains the timestamps (in
   milliseconds) or frame numbers matching the images in `'frames/images'`.
   Shape is the same.

5. `dataset['frames/images_path']` - contains the paths.


## Loading a previous dataset

After creating your dataset, you can (AND SHOULD!) load the previous dataset as
long as you don't wish to change your generator somehow.

There are two methods for reproducing a previous dataset:

1. `DatasetCreator.load_latest_dataset('my_dataset')`

2. `datasets_df = dataset_creator.get_registered_datasets('my_dataset')`

While the first method is pretty self-explanatory, the second returns a
DataFrame, containing the timestamps for all previous datasets with the
requested name. To reproduce a dataset given that:

```python
creator = DatasetCreator(creation_time=datasets_df['creation_time'][-2])
```

Please note that a reproduced dataset is not saved again for reproducibility, so
it is VERY recommended to load a previous dataset instead of creating a new
dataset with the same original generator. This will both save you the time
waiting for your dataset to be saved, and save the space in spanner.

## Available features

The following list is updated to 2023-06-06:

### ImagesFeature

A feature which reads images from a video / image / storage / etc... The basis
for this feature is provided by `video_io.AbstractVideoReader`. By passing
different types of readers (Current readers are `video_io.VideoFileReader`,
`video_io.ImagesReader`, `video_io.IngestedSstableReader`).

An example to an instantiation of an ImagesFeature is given:

```python
from dataset_creator.dataset_creator.features import images_feature

feature = images_feature.ImagesFeature(
  reader=video_io.VideoFileReader('/path/to/video.mp4'),
  # Also possible images_feature.READ_BY_FRAME_NUMBERS
  read_by=images_feature.READ_BY_TIMESTAMP_MILLIS,
  read_at=[timestamp_in_milliseconds],
  image_size=(224, 224)
)
```

### InferenceFeature

A feature which extracts `tf.Tensor` feature vectors from a pretrained model.
When instantiating this feature, a user must provide 3 arguments:

1. `keras_model_path` - A path to a saved Keras model. Please note that saved
   keras models are saved using `model.save('/path/to/saved/dir/')` and *NOT* by
   calling `tf.saved_model.save(model, '/path/to/saved/dir/')`.
2. `outputs_layer_names` - A sequence of layer names whose outputs the user
   wishes to extract.
3. `container_to_inputs` - A *STATELESS* function which converts an Example to
   an input to the model. This function must be stateless for reproducibility.
   This means that it cannot rely on outer imports, variables that are not
   defined in the function's body, and so on..

The population of example features is ordered by the order they appeared in the
original example. This can help easily write the `container_to_inputs` function.
For example:

```python
def example_to_inputs(example: example_lib.Example) -> list[np.ndarray]:
  if 'frame/images' not in example:
    return []
  return [example['frame/images']]
```

Even though the `'frame/images'` has not been yet populated at the time of
generation, since you can rely on the fact that population of an inference
feature will come after the population of an earlier images feature, this will
still work.

### MultiLeadSignalFeature

A feature which reads multi-lead signals, be it EEG, ECG, or other. The feature
is based on `base_signal_reader.AbstractMultiLeadSignalReader`. Current
implemented readers are:

1. `EdfSignalReader`: To read signals saved in .edf and .bdf formats.
2. `BrainVisionSignalReader`: To read signals saved in .vhdr and .eeg formats.
3. `EEGLABSignalReader`: To read signals saved in .set and .fdt formats.

Based on these readers, this feature also supports normalization of the signals
based on the following attributes:

1. `resample_at`: Regardless of the original sampling frequency of the signal,
   resample the signal after reading it as if it was originally sampled with
   this frequency.
2. `label_patterns`: A sequence of regex strings. If provided, only channels
   that match these patterns are included in the output signal, and channels
   return in the order matching the given patterns. This helps making sure that
   channel 0, 1, ... always correspond to the same semantic channels.

In addition to these attributes, the feature also provides the ability to only
read parts of the signal, using `start` and `end`.

IMPORTANT NOTE: `start` and `end` always refer to sample numbers of the original
signal. When using them in conjunction with `resample_at`, it's important to
note that the signal length will NOT BE `end - start`, as the signal is first
cropped, and only then resampled.

### LambdaFeature

A feature which adds features to an example according to a given stateless
function. This feature does not support parallelization of parts, as the user
only provides a single stateless function that operates on the example as a
whole.

As an example, please see the stateless function below which extracts landmarks
from an example

```python
def extract_landmarks(example):
  import mediapipe as mp
  import numpy as np
  import more_itertools

  from dataset_creator.video_io import video_io
  from mediapipe.tasks.python.core import base_options
  from mediapipe.tasks.python.vision import face_landmarker
  from mediapipe.tasks.python.vision.core import vision_task_running_mode

  ASSET_PATH = '/tmp/face_landmarker_v2.task'

  if 'images/images' not in example:
    return {
        'landmarks': np.zeros(1, dtype=np.float32)
    }

  images = example['images/images']  # (N, H, W, C)
  frame_numbers = example['images/images_read_at']
  with video_io.VideoFileReader(example['images/images_path']) as reader:
    fps = reader.fps
  timestamps = [
      int(1000 * frame_number / fps) for frame_number in frame_numbers
  ]

  with gfile.Open(ASSET_PATH, 'rb') as f:
    options = face_landmarker.FaceLandmarkerOptions(
        base_options.BaseOptions(model_asset_buffer=f.read()),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = face_landmarker.FaceLandmarker.create_from_options(options)

  images_landmarks = []
  for image, timestamp_millis in zip(images, timestamps):
    H, W, _ = image.shape
    image_landmarks = []
    results = landmarker.detect_for_video(image, timestamp_millis)
    for landmark in more_itertools.first(results.face_landmarks):
      image_landmarks.append(
          (landmark.x * H, landmark.y * W, landmark.z * W)
      )
    images_landmarks.append(image_landmarks)
  return {
      'landmarks': np.array(images_landmarks).astype(np.float32) # (N, 478, 3)
  }

feature = lambda_feature.LambdaFeature(extract_landmarks)
```

Please note 3 main points:

1. The user function MUST take care of the edge case where dependent features
   (such as `'images/images'` in the case above) don't exist. It must return an
   object with the proper type. This is because validation of the statelessness
   of the function takes place with mock values, and expects it to finish
   successfully.
2. The user function MUST return a mapping with `str` keys and valid features
   as values (see above). These output features are added to the example when
   the `LambdaFeature` is processed.
3. The function MUST not rely on any outer import, global variable, etc, to
   make sure it's stateless (and thus reproducible).
