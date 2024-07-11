"""Tests for example_bank_shard_dataset.py."""

from typing import Any, Callable

from absl.testing import parameterized  # type: ignore[import]

from dataset_creator import example_bank_shard_dataset
from dataset_creator import helpers
from dataset_creator import test_utils

# pylint: disable=protected-access


class ExampleBankShardDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._shards = helpers.glob_prefix(test_utils.mock_example_bank_prefix())

  @parameterized.named_parameters(
      ('torch', example_bank_shard_dataset.get_torch_dataset),
      ('tf', example_bank_shard_dataset.get_tf_dataset),
  )
  def test_dataset_length(self, get_dataset_fn: Callable[[str], Any]):
    dataset = [
        record for shard_path in self._shards
        for record in get_dataset_fn(shard_path)
    ]
    self.assertLen(dataset, 5)

  def test_torch_dataset_supports_random_access(self):
    dataset = example_bank_shard_dataset.get_torch_dataset(self._shards[0])
    self.assertIsInstance(dataset[1], bytes)

  def test_torch_dataset_raises_indexerror_on_out_of_bounds_index(self):
    dataset = example_bank_shard_dataset.get_torch_dataset(self._shards[0])
    with self.assertRaisesRegex(IndexError, 'invalid index'):
      _ = dataset[len(dataset)]


# pylint: enable=protected-access
