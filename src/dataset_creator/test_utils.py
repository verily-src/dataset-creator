"""A utility containing common functions to be used by tests."""

import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import tensorflow as tf
from typing_extensions import TypeAlias

from dataset_creator import example_lib
from dataset_creator.features import images_feature
from dataset_creator.features import inference_feature
from dataset_creator.features.video_io import base_video_io

THIS_DIR = Path(__file__).parent

Example = example_lib.Example
TimestampsSequence: TypeAlias = base_video_io.TimestampsSequence
FrameSize: TypeAlias = base_video_io.FrameSize

IMAGES_FEATURE_NAME = 'images'
INFERENCE_FEATURE_NAME = 'inference'
LABELS_FEATURE_NAME = 'labels'


def get_examples_generator(
    video_path: Optional[str] = None,
    video_timestamps_sequences: Optional[Sequence[TimestampsSequence]] = None,
    image_path: Optional[str] = None,
    image_size: Optional[FrameSize] = None,
    keras_model_path: Optional[str] = None,
    outputs_layer_names: Optional[Sequence[str]] = None,
    example_to_inputs: Optional[Callable[[Example], Any]] = None,
    inference_only: bool = False,
) -> Iterable[Example]:
  """Yields examples built from the given parameters.

  Args:
    video_path: The path corresponding to the yielded examples.
    video_timestamps_sequences: Each sequence in this collection corresponds to
      a single Example to be yielded from video_path.
    image_path: A path to an image used to yield an additional Example
      corresponding to the single frame contained in the image.
    image_size: The size to resize images to.
    keras_model_path: A path to a mock tf.keras.Model to be used in inference.
    outputs_layer_names: The names of layers to be inferences using the model.
    example_to_inputs: A callable to convert an Example to inputs of the model.
    inference_only: Only use the images for inference. In that case, the images
      are marked to be dropped.
  """
  if not video_timestamps_sequences:
    video_timestamps_sequences = default_video_timestamps_sequences()
  video_paths = [
      video_path or mock_video_path()
      for _ in video_timestamps_sequences
  ]
  video_timestamps_sequences = list(video_timestamps_sequences)

  image_path = image_path or mock_image_path()
  video_timestamps_sequences.append(timestamps_sequence_for_image())
  video_paths.append(image_path)

  inference_args = [keras_model_path, outputs_layer_names, example_to_inputs]
  if any(inference_args):
    if (
        keras_model_path is None or
        not outputs_layer_names or
        not example_to_inputs
    ):
      raise ValueError('Either all or no inference arguments must be provided.')
    inference_features = {
        INFERENCE_FEATURE_NAME: inference_feature.InferenceFeature(
            keras_model_path=keras_model_path,
            outputs_layer_names=outputs_layer_names,
            container_to_inputs=example_to_inputs,  # type: ignore[arg-type]
        )
    }
  else:
    inference_features = {}

  for sequence, path in zip(video_timestamps_sequences, video_paths):
    reader = images_feature.get_default_reader(path)
    yield Example({
        IMAGES_FEATURE_NAME: images_feature.ImagesFeature(
            reader=reader,
            read_by=images_feature.READ_BY_TIMESTAMP_MILLIS,
            read_at=sequence,
            image_size=image_size,
            drop_on_finalize=inference_only,
        ),
        LABELS_FEATURE_NAME: get_labels_from_sequence(sequence),
    } | inference_features)


def default_video_timestamps_sequences() -> Sequence[TimestampsSequence]:
  """Returns a default sequence of TimestampsSequences for testing."""
  return [[30], [100], [472], [51]]


def timestamps_sequence_for_image() -> TimestampsSequence:
  return [0]


def _example_to_inputs(example: Example) -> list[np.ndarray]:
  images_key = 'images/images'
  if images_key not in example:
    return []
  frame = example[images_key][0]  # type: ignore[index]
  return [frame.reshape(1, frame.size)]  # type: ignore[union-attr]


def inference_parameters(
    input_size: int = 10,
) -> tuple[str, Sequence[str], Callable[[Example], Any]]:
  """Returns parameters for mock inference."""
  model = tf.keras.Sequential(
      [
          tf.keras.layers.Input(input_size),
          tf.keras.layers.Dense(2, name='dense')
      ]
  )
  # pylint: disable=consider-using-with
  keras_model_path = os.path.join(
      tempfile.TemporaryDirectory().name, 'keras_model'
  )
  # pylint: enable=consider-using-with
  model.save(keras_model_path)
  return keras_model_path, ['dense'], _example_to_inputs


def mock_video_path() -> str:
  """Returns a default video_path for testing."""
  return str(THIS_DIR / 'features/video_io/testdata/test_video.mp4')


def mock_example_bank_prefix() -> str:
  """Returns the prefix to the test SSTable."""
  return str(THIS_DIR / 'testdata/storage-test')


def mock_image_path() -> str:
  """Returns a default image_path for testing."""
  return str(THIS_DIR / 'features/video_io/testdata/frame_ts_60s.jpg')


def get_labels_from_sequence(sequence: TimestampsSequence) -> list[int]:
  """A mock label extractor based on the Example's timestamps."""
  return [timestamp % 10 for timestamp in sequence]
