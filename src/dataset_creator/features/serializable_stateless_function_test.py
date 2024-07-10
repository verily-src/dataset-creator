"""Tests for serializable_stateless_function."""

import functools
from typing import Any, Callable

from absl.testing import parameterized  # type: ignore[import]
import tensorflow as tf

from dataset_creator.features import serializable_stateless_function

SerializableStatelessFunction = (
    serializable_stateless_function.SerializableStatelessFunction
)


def foo(x: Any) -> Any:
  return x


def internal_import(x: Any) -> Any:
  import numpy as np  # pylint: disable=import-outside-toplevel
  return np.array(x)


def outer(y):
  def inner(x):
    return x
  return inner(y)


recursive_factorial: Callable[[int], int] = (
    lambda n: 1 if n == 0 else n * recursive_factorial(n - 1)
)


VALID_TEST_CASES = (
    ('lambda', lambda x: x),
    ('function', foo),
    ('internal_import', internal_import),
    ('nested', outer),
)

INVALID_TEST_CASES = (
    ('assumes_import', lambda x: tf.add(x, 1)),
    ('assumes_global', lambda x: foo(x + 1)),
    ('instance_method', str().splitlines),  # depends on the instance.
    ('partial', functools.partial(lambda x, y: x + y, y=1)),
    # Recursive is also stateful, since it relies on the specific function name
    ('recursive', recursive_factorial),
)


class SerializableStatelessFunctionTest(parameterized.TestCase):

  @parameterized.named_parameters(*VALID_TEST_CASES)
  def test_serialize(self, fn: Callable[[Any], Any]):
    self.assertNotEmpty(SerializableStatelessFunction(fn).serialize())

  @parameterized.named_parameters(*VALID_TEST_CASES)
  def test_deserialize_results_in_same_function(self, fn: Callable[[Any], Any]):
    serialized = SerializableStatelessFunction(fn).serialize()
    deserialized_fn = SerializableStatelessFunction.deserialize(serialized)
    self.assertEqual(deserialized_fn(1), fn(1))

  @parameterized.named_parameters(*INVALID_TEST_CASES)
  def test_validate_raises_on_invalid(self, fn: Callable[[Any], Any]):
    with self.assertRaises(ValueError):
      SerializableStatelessFunction(fn).validate([1])

  @parameterized.named_parameters(*VALID_TEST_CASES)
  def test_validate_does_not_raise_on_valid(self, fn: Callable[[Any], Any]):
    SerializableStatelessFunction(fn).validate([1])

  def test_validate_with_arguments_that_assume_some_context(self):
    arg = tf.constant(1)
    SerializableStatelessFunction(lambda tensor: tensor.shape).validate([arg])
    with self.assertRaises(ValueError):
      SerializableStatelessFunction(
          lambda tensor: tf.add(tensor, 1)
      ).validate([arg])
