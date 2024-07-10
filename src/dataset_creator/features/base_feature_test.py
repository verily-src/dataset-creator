"""Tests for base_feature.py."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from absl.testing import parameterized  # type: ignore[import]
import numpy as np

from dataset_creator.features import base_feature
from dataset_creator.features import serializable

# pylint: disable=protected-access

class MockSerializable(serializable.Serializable):
  """A mock of a Serializable object."""

  def __init__(self, num: int):
    self._num = num

  def serialize(self) -> bytes:
    return str(self._num).encode()

  @classmethod
  def deserialize(cls, serialized: bytes) -> MockSerializable:
    return cls(int(serialized.decode()))


class MockFeature(base_feature.CustomFeature):
  """A mock of a CustomFeature object."""

  def split(self) -> Sequence[int]:
    return [0]

  def process(self, metadata_value: int, _) -> int:
    return metadata_value

  def merge(
      self,
      values: Sequence[base_feature.ValueFeature]
  ) -> Mapping[str, base_feature.ValueFeature]:
    return {'values': sum(values)}  # type: ignore[arg-type]

  @property
  def output_signature(self) -> Mapping[str, Any]:
    return {'values': int}


class ValidMockFeature(MockFeature):
  """A mock of a VALID CustomFeature."""

  _serializable_classes = [MockSerializable]

  def __init__(
      self,
      a: int,
      b: Sequence[str],
      serializable_value: MockSerializable,
      **kwargs
  ):
    super().__init__(
        a=a,
        b=b,
        serializable_value=serializable_value,
        **kwargs
    )
    self._a = a
    self._b = b
    self._serializable_value = serializable_value


class CustomFeatureTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('not_serializable', np.array([1])),
      ('serializable_not_declared_in_serializables', MockSerializable(1)),
      ('invalid_value', [[1]]),
  )
  def test_constructor_raises_valueerror(self, init_value: Any):
    with self.assertRaises(ValueError):
      MockFeature(value=init_value)

  def test_mock_feature_works_as_expected(self):
    feature = MockFeature()
    context = feature.create_context()  # pylint: disable=assignment-from-none
    merged_outputs = feature.merge(
        [feature.process(value, context) for value in feature.split()]
    )
    self.assertSameElements(feature.output_signature, merged_outputs)

  def test_get_config(self):
    feature = ValidMockFeature(
        a=1,
        b=['1', '2', '3'],
        serializable_value=MockSerializable(0),
        drop_on_finalize=False,
    )
    expected_metadata = {
        f'a@{base_feature._PRIMITIVE}': 1,
        f'b@{base_feature._PRIMITIVE}': ['1', '2', '3'],
        f'drop_on_finalize@{base_feature._PRIMITIVE}': False,
        'serializable_value@MockSerializable': b'0',
    }
    self.assertSameStructure(feature.get_config(), expected_metadata)

  @parameterized.named_parameters(
      ('normal_sequence', ['1', '2', '3']),
      ('empty_sequence', []),
  )
  def test_from_config(self, b: Sequence[str]):
    config = {
        f'a@{base_feature._PRIMITIVE}': 1,
        f'b@{base_feature._PRIMITIVE}': b,
        'serializable_value@MockSerializable': b'0'
    }
    feature = ValidMockFeature.from_config(config)  # type: ignore[arg-type]
    self.assertIsInstance(feature, ValidMockFeature)
    self.assertEqual(feature._a, 1)
    self.assertEqual(feature._b, b)
    self.assertEqual(feature._serializable_value._num, 0)

  def test_from_config_raises_with_unknown_serializable_class(self):
    config = {
        f'a@{base_feature._PRIMITIVE}': 1,
        'serializable_value@MockSerializable': b'0'
    }
    with self.assertRaisesRegex(
        ValueError, 'MockSerializable.*does not appear in _serializable_classes'
    ):
      # MockFeature does not contain any _serializable_classes
      MockFeature.from_config(config)

# pylint: enable=protected-access
