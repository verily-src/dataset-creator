"""A module to get torch and tensorflow datasets from example bank shards.

This module offers 2 APIs to get datasets from an example bank shard:
  1. get_torch_dataset(shard_path) -> torch.utils.data.Dataset, and:
  2. get_tf_dataset(shard_path) -> tf.data.Dataset.
"""
import functools

import riegeli  # type: ignore[import]
from riegeli.tensorflow.ops import riegeli_dataset_ops  # type: ignore[import]
import tensorflow as tf
import torch  # type: ignore[import]

from dataset_creator.pipeline import example_bank_sink


class _ExampleBankShardDataset(torch.utils.data.Dataset):
  """A reader that reads records from a shard."""

  def __init__(self, shard_path: str):
    self._shard_path = shard_path

  @functools.cached_property
  def _positions(self) -> dict[int, riegeli.RecordPosition]:
    positions_path = example_bank_sink.get_positions_file_path(self._shard_path)
    with open(positions_path, 'rb') as f:
      with riegeli.RecordReader(f) as reader:
        positions = map(
            riegeli.RecordPosition.from_bytes, reader.read_records()
        )
        return dict(enumerate(positions))

  def __len__(self) -> int:
    """Returns the number of records saved in this shard."""
    return len(self._positions)

  def __getitem__(self, index: int) -> bytes:
    """Reads the record with given index from this shard.
    
    Args:
      index: The index to read.
    
    Returns:
      The record at index.
    """
    if index >= len(self):
      raise IndexError('An invalid index was given.')
    with open(self._shard_path, 'rb') as f:
      with riegeli.RecordReader(f) as reader:
        reader.seek(self._positions[index])
        return reader.read_record()


def get_torch_dataset(shard_path: str) -> _ExampleBankShardDataset:
  """Returns a dataset that its elements are records of this shard.
  
  Args:
    shard_path: The path of the shard whose dataset is returned.
  """
  return _ExampleBankShardDataset(shard_path)


def get_tf_dataset(shard_path: str) -> tf.data.Dataset:
  """Returns a dataset that its elements are records of this shard.
  
  Args:
    shard_path: The path of the shard whose dataset is returned.
  """
  return riegeli_dataset_ops.RiegeliDataset(shard_path)
