"""Tests for test_utils."""

import re

from absl.testing import absltest  # type: ignore[import]
import more_itertools
import numpy as np
import tensorflow as tf

from dataset_creator import example_lib
from dataset_creator import test_utils
from dataset_creator.features import fields

# pylint: disable=protected-access

def get_full_layer_name_from_prefix(
    model: tf.keras.Model,
    layer_name: str
) -> str:
  """Returns the full layer name from the model.

  This method is needed since Tensorflow might add _{number} to the end of layer
  names upon loading.

  Args:
    model: The model to search for the layer in.
    layer_name: The layer name prefix to be found.

  Raises:
    ValueError: In any case layer_name does not match a single layer in model.
  """
  return more_itertools.one(
      layer.name for layer in model.layers
      if re.fullmatch(f'{layer_name}(_[0-9]*)?', layer.name) is not None)


class TestUtilsTest(absltest.TestCase):

  def test_default_timestamps_sequences_are_not_sorted(self):
    timestamps_sequences = test_utils.default_video_timestamps_sequences()
    self.assertNotEqual(sorted(timestamps_sequences), timestamps_sequences)

  def test_examples_generated_in_the_same_order_as_timestamps_sequences(self):
    timestamps_sequences = [[0], [1], [2], [3]]
    generated_timestamps_sequences = [
        example[test_utils.IMAGES_FEATURE_NAME]._read_at
        for example in test_utils.get_examples_generator(
            video_timestamps_sequences=timestamps_sequences)
    ]
    image_timestamps = test_utils.timestamps_sequence_for_image()
    self.assertEqual(timestamps_sequences + [image_timestamps],
                     generated_timestamps_sequences)

  def test_inference_parameters(self):
    frame = np.zeros((224, 224, 3), dtype=np.float32)
    images_key = example_lib.nested_key(
        test_utils.IMAGES_FEATURE_NAME, fields.IMAGES
    )
    mock_example = example_lib.Example({images_key: frame[None, ...]})
    keras_model_path, outputs_layer_names, example_to_inputs = (
        test_utils.inference_parameters(frame.size)
    )
    model = tf.keras.models.load_model(keras_model_path)
    for layer_name in outputs_layer_names:
      get_full_layer_name_from_prefix(model, layer_name)
    for model_input in example_to_inputs(mock_example):
      model(model_input)

  def test_get_examples_generator_raises_on_partial_inference_params(self):
    model_path, *_ = test_utils.inference_parameters()
    with self.assertRaisesRegex(
        ValueError, 'Either all or no inference arguments must be provided'
    ):
      list(test_utils.get_examples_generator(keras_model_path=model_path))

# pylint: enable=protected-access
