"""Tests for serializable.py."""

from __future__ import annotations

import pickle

from absl.testing import absltest  # type: ignore[import]

from dataset_creator.features import serializable


class Mock(serializable.Serializable):
  def __init__(self, value: int):
    self.value = value

  def serialize(self) -> bytes:
    return bytes(self.value)

  @classmethod
  def deserialize(cls, serialized: bytes) -> Mock:
    return cls(len(serialized))

class SerializableStatelessFunctionTest(absltest.TestCase):

  def test_pickle_compliance(self):
    mock = Mock(5)
    restored_mock = pickle.loads(pickle.dumps(mock))
    self.assertEqual(restored_mock.value, mock.value)
