"""Tests for inference_feature.py."""

import tempfile

import tensorflow as tf

from dataset_creator.features import inference_feature
from dataset_creator.features import serializable_stateless_function

# pylint: disable=protected-access

_SerializableStatelessFn = (
    serializable_stateless_function.SerializableStatelessFunction
)


class InferenceFeatureTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model = tf.keras.Sequential([
        tf.keras.layers.Input((4,)),
        tf.keras.layers.Lambda(lambda x: x + 1, name='add'),
        tf.keras.layers.Dense(
            2, name='dense', use_bias=False, kernel_initializer='zeros'
        ),
    ])
    # pylint: disable-next=consider-using-with
    keras_model_path = tempfile.TemporaryDirectory().name
    self.model.save(keras_model_path)
    self.outputs_layer_names = ['add', 'dense']

    def container_to_inputs(_):
      import tensorflow  # pylint:disable=import-outside-toplevel,reimported
      return [i * tensorflow.ones((1, 4)) for i in range(3)]

    self.feature = inference_feature.InferenceFeature(
        keras_model_path=keras_model_path,
        outputs_layer_names=self.outputs_layer_names,
        container_to_inputs=container_to_inputs,
    )
    self.feature.container = {'test': self.feature}

  def test_isnt_self_contained(self):
    self.assertFalse(self.feature.is_self_contained)

  def test_split_outputs_are_compatible_with_model(self):
    for model_input in self.feature.split():
      self.model(model_input, training=False)

  def test_created_context_model_has_outputs_layer_names_keys(self):
    inference_model = self.feature.create_context()
    self.assertSameElements(
        inference_model.output.keys(), self.outputs_layer_names
    )

  def test_process_invokes_inference_model_on_input(self):
    model = self.feature.create_context()
    processed = self.feature.process(tf.ones((1, 4)), model)
    self.assertSameElements(processed.keys(), self.outputs_layer_names)
    self.assertAllEqual(processed['add'], tf.fill((1, 4), 2))
    self.assertAllEqual(processed['dense'], tf.zeros((1, 2)))

  def test_merge_concatenates_sequence_of_process_outputs(self):
    processed_values = [{'test': i * tf.ones((1, 10))} for i in range(3)]
    merged_outputs = self.feature.merge(processed_values)
    self.assertAllEqual(
        merged_outputs['test'],
        tf.convert_to_tensor([[0] * 10, [1] * 10, [2] * 10]),
    )

  def test_create_context_raises_with_stateful_container_to_inputs(self):
    def stateful_fn(_):
      return [i * tf.ones((1, 4)) for i in range(3)]
    self.feature._container_to_inputs = (
        _SerializableStatelessFn(stateful_fn)
    )
    with self.assertRaises(ValueError):
      self.feature.create_context()

  def test_create_context_raises_with_non_sequential_container_to_inputs(self):
    self.feature._container_to_inputs = (
        _SerializableStatelessFn(lambda _: 3)
    )
    with self.assertRaisesRegex(ValueError, 'must be a sequence'):
      self.feature.create_context()

# pylint: enable=protected-access
