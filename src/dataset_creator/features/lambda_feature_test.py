"""Tests for lambda_feature.py."""

import tensorflow as tf

from dataset_creator.features import lambda_feature


def _container_to_outputs(_) -> dict[str, lambda_feature._ValueFeature]:
  import numpy  # pylint: disable=import-outside-toplevel, reimported
  import tensorflow  # pylint: disable=import-outside-toplevel, reimported
  return {
      'primitive_output': True,
      'tf_output': tensorflow.constant([1., 2., 3.]),
      'np_output': numpy.array([1., 2., 3.]),
      'sequence_of_primitives': [b'1', b'2', b'3'],
      'empty_sequence': [],
      'none': None,
  }


class LambdaFeatureTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.feature = lambda_feature.LambdaFeature(_container_to_outputs)

  def test_split_returns_a_nonempty_sequence(self):
    self.assertNotEmpty(self.feature.split())

  def test_process_doesnt_return_none(self):
    self.assertIsNotNone(self.feature.process(None, None))

  def test_merge_returns_the_expected_output(self):
    context = self.feature.create_context()
    merged = self.feature.merge(
        [self.feature.process(value, context) for value in self.feature.split()]
    )
    expected = _container_to_outputs(None)
    self.assertAllClose(expected.pop('np_output'), merged.pop('np_output'))
    self.assertAllClose(expected.pop('tf_output'), merged.pop('tf_output'))
    self.assertEqual(expected, merged)

  def test_lambda_feature_reconstructible_from_config(self):
    config = self.feature.get_config()
    self.assertIsNotNone(lambda_feature.LambdaFeature.from_config(config))

  def test_lambda_feature_raises_without_a_lambda_function(self):
    with self.assertRaisesRegex(ValueError, 'lambda_fn must be provided.'):
      lambda_feature.LambdaFeature()
