"""Tests for generic_lambda_feature.py."""

from typing import Any, Mapping, Sequence

from absl.testing import parameterized  # type: ignore[import]
import numpy as np
import numpy.typing as np_typing  # pylint: disable=unused-import
import tensorflow as tf

from dataset_creator.features import generic_lambda_feature


def _container_to_outputs(*_) -> Mapping[str, Any]:
  import numpy  # pylint: disable=import-outside-toplevel, reimported
  import tensorflow  # pylint: disable=import-outside-toplevel, reimported
  return {
      'primitive_output': True,
      'tf_output': tensorflow.constant([1., 2., 3.]),
      'np_output': numpy.array([1., 2., 3.], dtype=np.float32),
      'sequence_of_primitives': [b'1', b'2', b'3'],
      'empty_sequence': [],
      'none': None,
  }


class GenericLambdaFeatureTest(parameterized.TestCase):

  def test_output_signature(self):
    feature = generic_lambda_feature.GenericLambdaFeature(
        split_fn=lambda container: [container],
        process_fn=_container_to_outputs,
        merge_fn=lambda values: values[0],
    )
    self.assertEqual(
        {
            'primitive_output': bool,
            'tf_output': tf.Tensor,
            'np_output': np.typing.NDArray[np.float32],
            'sequence_of_primitives': Sequence[bytes],
            'empty_sequence': Sequence[float],
            'none': type(None),
        },
        feature.output_signature
    )

  @parameterized.named_parameters(
      (
          'invalid_split',
          {
              'split_fn': lambda _: np.zeros(5),
              'process_fn': lambda *_: 0,
              'merge_fn': lambda _: {},
          }
      ),
      (
          'invalid_process',
          {
              'split_fn': [0, 1, 2],
              'process_fn': lambda value, _: np.array(value),
              'merge_fn': lambda _: {},
          }
      ),
      (
          'invalid_process_without_context',
          {
              'split_fn': [0, 1, 2],
              'process_fn': lambda value: np.array(value + 1),
              'merge_fn': lambda _: {},
              'process_with_context': False,
          }
      ),
      (
          'invalid_merge',
          {
              'split_fn': [0, 1, 2],
              'process_fn': lambda value, _: value,
              'merge_fn': lambda values: {'test': np.array(values)},
          }
      ),
  )
  def test_create_context_raises_on_stateful_functions(
      self, feature_kwargs: Mapping[str, Any]
  ):
    feature = generic_lambda_feature.GenericLambdaFeature(**feature_kwargs)
    with self.assertRaises(ValueError):
      feature.create_context()

  @parameterized.named_parameters(
      ('with_context', True), ('without_context', False)
  )
  def test_process(self, with_context: bool):
    if with_context:
      process_fn: generic_lambda_feature._ProcessFn = lambda x, _: x
    else:
      def process_fn(x):
        return x
    feature = generic_lambda_feature.GenericLambdaFeature(
        split_fn=lambda _: [0, 1, 2],
        process_fn=process_fn,
        merge_fn=lambda _: {},
        process_with_context=with_context
    )
    # Validate that in both cases, create_context does not raise.
    context = feature.create_context()
    self.assertEqual(feature.process(0, context), 0)
