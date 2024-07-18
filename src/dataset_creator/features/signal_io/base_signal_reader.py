"""Abstract class for SignalReaders."""

from __future__ import annotations

import abc
import collections
import dataclasses
import datetime
import enum
import functools
import os
from typing import Optional, Sequence

import numpy as np

from dataset_creator.features import serializable


class PhysicalUnit(enum.Enum):
  UNKNOWN = 1.
  VOLT = 1.
  MILLI_VOLT = 1e-3
  MICRO_VOLT = 1e-6
  NANO_VOLT = 1e-9


STRING_TO_PHYSICAL_UNIT: collections.defaultdict[str, PhysicalUnit] = (
    collections.defaultdict(lambda: PhysicalUnit.UNKNOWN)
)
STRING_TO_PHYSICAL_UNIT.update(
    {
        'V': PhysicalUnit.VOLT,
        'mV': PhysicalUnit.MILLI_VOLT,
        # µ has 2 possible unicode characters, so be ready to use both.
        'μV': PhysicalUnit.MICRO_VOLT,  # b'\xce\xbcV'
        'µV': PhysicalUnit.MICRO_VOLT,  # b'\xc2\xb5V'
        'uV': PhysicalUnit.MICRO_VOLT,
        'nV': PhysicalUnit.NANO_VOLT,
    }
)


@dataclasses.dataclass
class LeadMetadata:
  lead_num: int
  label: str
  physical_dim: PhysicalUnit


class AbstractMultiLeadSignalReader(serializable.Serializable, abc.ABC):
  """An EDF reader which supports CNS paths, and can be reused."""

  def __init__(
      self,
      path: str, check_path: bool = True,
      allowed_extensions: Sequence[str] = (),
  ):
    """Instantiates a new reader.

    Args:
      path: The path to the multi lead signal. Could be either a file or a
        directory.
      check_path: If True, checks that the given path exists.
      allowed_extensions: A list of allowed extensions for the path.

    Attributes:
      path: The given path.

    Raises:
      FileNotFoundError: In case of check_path and the file doesn't exist.
      ValueError: In case the file format is not allowed.
    """
    self.path = path
    self._check_path = check_path

    if check_path and not os.path.exists(path):
      raise FileNotFoundError(f'The path {path} cannot be found.')
    _, ext = os.path.splitext(path)
    if allowed_extensions and ext.lower() not in allowed_extensions:
      raise ValueError(f'Invalid file format for {self.__class__}: {ext}.')

  @functools.cached_property
  @abc.abstractmethod
  def num_leads(self) -> int:
    """Returns the number of leads in self.path."""

  @functools.cached_property
  @abc.abstractmethod
  def lead_headers(self) -> Sequence[LeadMetadata]:
    """Returns a sequence of per-lead metadata."""

  @functools.cached_property
  @abc.abstractmethod
  def sampling_frequency(self) -> float:
    """Returns the sampling frequency of the signal."""

  @property
  @abc.abstractmethod
  def total_samples(self) -> int:
    """Returns the number of samples per channel in the signal."""

  @property
  def duration(self) -> datetime.timedelta:
    """Returns the total length of the signal in seconds."""
    return datetime.timedelta(
        seconds=self.total_samples / self.sampling_frequency
    )

  def read_signal(
      self, lead_num: int, start: int = 0, end: int = -1
  ) -> np.ndarray | None:
    """Reads the signal from the given channel from self._edf_path.

    Args:
      lead_num: The number of lead to read.
      start: The 0-based sample number to start at.
      end: The 0-based end sample to end at (exclusive). If <0, reads until the
        end of the signal. Default is -1.

    Returns:
      The signal for the given lead.
    """
    if lead_num < 0 or lead_num >= self.num_leads:
      return None
    if end < 0 or end > self.total_samples:
      end = self.total_samples
    if start < 0 or end > self.total_samples or start >= end:
      return None
    return self._read_signal(lead_num, start, end)

  @abc.abstractmethod
  def _read_signal(
      self, lead_num: int, start: int, end: int
  ) -> Optional[np.ndarray]:
    """Reads the signal from the given channel from self._edf_path."""

  def serialize(self) -> bytes:
    return f'{self.path},{self._check_path}'.encode()

  @classmethod
  def deserialize(cls, serialized: bytes) -> AbstractMultiLeadSignalReader:
    path, check_path = serialized.decode().split(',')
    return cls(path, check_path=check_path == 'True')
