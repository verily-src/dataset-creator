"""Tests for example_lib.py."""

from __future__ import annotations

import pickle
from typing import Sequence

from absl.testing import parameterized  # type: ignore[import]
import numpy as np
import tensorflow as tf

from dataset_creator import example_lib
from dataset_creator.features import base_feature
from dataset_creator.features import generic_typing
from dataset_creator.features import lambda_feature

# pylint: disable=protected-access

class MockFeature(base_feature.CustomFeature):
  """A mock for a CustomFeature."""

  def __init__(self, drop_on_finalize: bool = False, **kwargs: int):
    super().__init__(drop_on_finalize=drop_on_finalize, **kwargs)
    self.values = kwargs.values()

  def split(self) -> Sequence[int]:
    return list(self.values)

  def process(self, metadata_value: int, _) -> int:
    return metadata_value + 1

  def merge(
      self, values: Sequence[int]) -> dict[str, base_feature.PrimitiveType]:
    return {str(i): value for i, value in enumerate(values)} | {
        'test_to_exclude': 1,
        'test_to_include': [1, 2, 3]
    }

  @property
  def output_signature(self):
    return {str(i): int for i in range(len(self.values))} | {
        'test_to_include': Sequence[int]
    }


class UnknownFeature(MockFeature):
  """This feature is not in example_lib._TYPE_NAME_TO_TYPE."""


class InvalidFeature(MockFeature):
  """This feature contains invalid characters in its output_signature."""
  @property
  def output_signature(self):
    return {example_lib._TF_EXAMPLE_SEPARATOR_CHAR: int}

class UnknownType:
  pass


def setUpModule():
  example_lib._TYPE_NAME_TO_TYPE['MockFeature'] = MockFeature
  example_lib._TYPE_NAME_TO_TYPE['InvalidFeature'] = InvalidFeature


class ExampleLibTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.feature = MockFeature(var=3)
    self.example = example_lib.Example({
        'int': 1,
        'float': 3.1,
        'bool': True,
        'none': None,
        'bytes': b'123',
        'str': '123',
        'primitive_sequence': ('a', 'b', 'c'),
        'single_element_list': [True],
        'uint8_np_array': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        'bool_np_array': np.array([[True, False]]),
        'string_np_array': np.array(['123', '4567']),
        'bytes_np_array': np.array([b'123', b'4567']),
        'tensor': tf.constant([[7., 8., 9.], [10., 11., 12.]]),
        'length_one_tensor': tf.zeros((1, 2), dtype=tf.float32),
        'feature': self.feature,
        'lambda': lambda_feature.LambdaFeature(lambda _: {'k': list(range(42))})
    })

  def assertEqualFeatures(
      self, a: base_feature.Feature, b: base_feature.Feature
  ):
    if a is None:
      self.assertIsNone(b)
    elif isinstance(a, float):
      self.assertAlmostEqual(a, b, places=5)
    elif generic_typing.generic_isinstance(a, base_feature.PrimitiveType):
      self.assertEqual(a, b)
    elif isinstance(a, base_feature.CustomFeature):
      self.assertIsInstance(b, type(a))
      self.assertSameElements(
          a._kwargs,  # type: ignore[union-attr]
          b._kwargs,  # type: ignore[union-attr]
      )
    # Now we are left with ndarrays / Tensors / Sequences of those.
    else:
      self.assertEqual(type(a), type(b))
      self.assertAllEqual(a, b)

  def assertSameExample(self, a: example_lib.Example, b: example_lib.Example):
    self.assertEqual(len(a.items()), len(b.items()))
    for (a_key, a_value), (b_key, b_value) in zip(a.items(), b.items()):
      self.assertEqual(a_key, b_key)
      self.assertEqualFeatures(a_value, b_value)

  def test_constructor_sets_container_for_base_features(self):
    feature = MockFeature()
    test_example = example_lib.Example({'test': feature})
    self.assertIs(test_example, feature.container)

  @parameterized.named_parameters(
      ('non_str_keys', {1: 2}),
      ('dictionary_feature', {'test': {1: 2}}),
      ('list_of_lists_feature', {'test': [[1, 2]]}),
      ('unknown_feature', {'test': UnknownFeature()}),
      ('unknown_type', {'test': UnknownType()}),
      ('list_of_np_arrays', {'test': [np.zeros(2)]}),
      ('list_of_dicts', {'test': [{'inner': 2}]}),
      ('empty_list', {'test': []}),
      ('invalid_key', {f'test{example_lib._TF_EXAMPLE_SEPARATOR_CHAR}': 1}),
      ('invalid_feature', {'test': InvalidFeature()}),
      ('np_array_with_invalid_dtype', {'test': np.array([], dtype=np.void)})
  )
  def test_invalid_example(self, features):
    with self.assertRaises(ValueError):
      example_lib.Example(features)

  def test_num_custom_features_of_example(self):
    feature = MockFeature()
    test_example = example_lib.Example({'test': feature})
    self.assertEqual(test_example.num_custom_features, 1)

  def test_from_config_on_get_config_output_returns_the_same_example(self):
    self.assertSameExample(
        self.example,
        example_lib.Example.from_config(self.example.get_config())
    )

  def test_to_bytes_returns_some_non_empty_string(self):
    self.assertNotEmpty(self.example.to_bytes())

  def test_from_bytes_returns_to_original_example(self):
    self.assertSameExample(
        self.example,
        example_lib.Example.from_bytes(self.example.to_bytes())
    )

  def _populate_example(self) -> example_lib.Example:
    custom_features_outputs: dict[str, base_feature.ValueFeature] = {}
    for name, feature in self.example.items():
      if isinstance(feature, base_feature.CustomFeature):
        for k, v in feature.merge(
            [feature.process(value, None) for value in feature.split()]
        ).items():
          custom_features_outputs[example_lib.nested_key(name, k)] = v
    return example_lib.Example(self.example | custom_features_outputs)

  def test_finalize_filters_the_outputs_correctly(self):
    populated_example = self._populate_example()
    finalized = populated_example.finalize()
    expected_keys_related_to_features = []
    for name, feature in self.example.items():
      if isinstance(feature, base_feature.CustomFeature):
        for signature_key in feature.output_signature:
          expected_keys_related_to_features.append(
              example_lib.nested_key(name, signature_key)
          )
    expected_keys_related_to_values = [
        user_key for user_key, feature in self.example.items()
        if not isinstance(feature, base_feature.CustomFeature)
    ]
    self.assertSameElements(
        expected_keys_related_to_values + expected_keys_related_to_features,
        finalized.keys(),
    )

  def test_tensors_that_arent_floats_raise_valueerror(self):
    value = tf.constant([False, True], dtype=tf.bool)
    with self.assertRaises(ValueError):
      example_lib.Example({'value': value})

  def test_custom_features_after_unpickling_contains_container_reference(self):
    example = example_lib.Example({'feature': self.feature})
    restored_example = pickle.loads(pickle.dumps(example))
    self.assertIn('feature', example.keys())
    self.assertIs(restored_example['feature'].container, restored_example)

  def test_get_finalized_tf_example_io_spec(self):
    finalized_io_spec = self.example.get_finalized_tf_example_io_spec()
    finalized_tf_example = self._populate_example().finalize().to_tf_example()

    parsed_tf_example = tf.io.parse_single_example(
        finalized_tf_example.SerializeToString(), finalized_io_spec
    )
    self.assertSameElements(
        finalized_tf_example.features.feature.keys(),
        parsed_tf_example.keys(),
    )

  @parameterized.named_parameters(('eager', True), ('graph_mode', False))
  def test_full_tf_example_parsing_matches_finalization(
      self, run_eagerly: bool
  ):
    tf.config.run_functions_eagerly(run_eagerly)
    io_spec = self.example.get_finalized_tf_example_io_spec()
    populated = self._populate_example()
    tf_example = tf.io.parse_single_example(populated.to_bytes(), io_spec)
    parsed = example_lib.normalize_parsed_tf_example(tf_example)
    finalized = populated.finalize()

    self.assertSameElements(finalized.keys(), parsed.keys())
    self.assertDTypeEqual(parsed['bool'], tf.bool)
    self.assertAllEqual(parsed['uint8_np_array'], finalized['uint8_np_array'])
    self.assertNotIsInstance(parsed['primitive_sequence'], tf.SparseTensor)

  def test_example_str_is_non_empty(self):
    self.assertNotEmpty(str(self.example))

  @parameterized.named_parameters(
      ('bool', np.array([True, False])),
      ('int', np.array([0, 1], dtype=int)),
      ('uint8', np.array([0, 1], dtype=np.uint8)),
      ('float64', np.array([1., 2.], dtype=np.float64)),
  )
  def test_example_restores_np_array_dtype_after_from_bytes(
      self, np_array: np.ndarray
  ):
    original_dtype = np_array.dtype
    example = example_lib.Example({'test': np_array})
    restored_example = example_lib.Example.from_bytes(example.to_bytes())
    self.assertDTypeEqual(restored_example['test'], original_dtype)

# pylint: enable=protected-access
