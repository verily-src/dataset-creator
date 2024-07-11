"""Implements an AbstractMultiLeadSignalReader for NetStation .raw files."""

import functools
import struct
from typing import Optional, Sequence

import numpy as np

from dataset_creator.features.signal_io import base_signal_reader
from dataset_creator.features.signal_io import utils

# Specification for the simple binary format can be found at:
# https://sccn.ucsd.edu/eeglab/testfiles/EGI/NEWTESTING/rawformat.pdf

_VERSION_OFFSET = 0
_SAMPLING_FREQUENCY_OFFSET = 20
_NUM_CHANNELS_OFFSET = 22
_CONVERSION_BITS_OFFSET = 26
_AMPLIFIER_RANGE_OFFSET = 28
_NUM_SAMPLES_OFFSET = 30
_NUM_EVENTS_OFFSET = 34
_HEADER_LENGTH_BEFORE_EVENTS = 36
_EVENT_HEADER_SIZE = 4

_ALLOWED_EXTENSIONS = ['.raw']

_AbstractReader = base_signal_reader.AbstractMultiLeadSignalReader


def _read_from_path(path: str, offset: int, fmt: str) -> int:
  with open(path, 'rb') as f:
    f.seek(offset)
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


class SimpleBinarySignalReader(_AbstractReader):
  """A reader for .raw files."""

  def __init__(self, path: str, **kwargs):
    kwargs['allowed_extensions'] = _ALLOWED_EXTENSIONS
    super().__init__(path, **kwargs)

  @functools.cached_property
  def num_leads(self) -> int:
    """See base class."""
    return _read_from_path(self.path, _NUM_CHANNELS_OFFSET, fmt='>H')

  @functools.cached_property
  def sampling_frequency(self) -> float:
    """See base class."""
    return _read_from_path(self.path, _SAMPLING_FREQUENCY_OFFSET, fmt='>H')

  @functools.cached_property
  def total_samples(self) -> int:
    """See base class."""
    return _read_from_path(self.path, _NUM_SAMPLES_OFFSET, fmt='>I')

  @functools.cached_property
  def _version(self) -> int:
    return _read_from_path(self.path, _VERSION_OFFSET, fmt='>I')

  @functools.cached_property
  def num_events(self) -> int:
    return _read_from_path(self.path, _NUM_EVENTS_OFFSET, fmt='>H')

  @functools.cached_property
  def _conversion_bits(self) -> int:
    return _read_from_path(self.path, _CONVERSION_BITS_OFFSET, fmt='>H')

  @functools.cached_property
  def _amplifier_range_in_volts(self) -> float:
    amplifier_range_in_millivolts = _read_from_path(
        self.path, _AMPLIFIER_RANGE_OFFSET, fmt='>H'
    )
    return (amplifier_range_in_millivolts / 1000) or 1

  @functools.cached_property
  def lead_headers(self) -> Sequence[base_signal_reader.LeadMetadata]:
    headers = []
    for i in range(self.num_leads):
      headers.append(
          base_signal_reader.LeadMetadata(
              lead_num=i,
              label=f'E{i}',  # .raw fiels don't contain channel names
              physical_dim=base_signal_reader.PhysicalUnit.MICRO_VOLT
          )
      )
    return headers

  def _read_signal(
      self, lead_num: int, start: int, end: int
  ) -> Optional[np.ndarray]:
    version_to_sample_size = {2: 2, 4: 4, 6: 8}
    fmt = 'f' if self._version > 2 else 'i'
    np_dtype = f'>{fmt}{version_to_sample_size[self._version]}'

    offset = _HEADER_LENGTH_BEFORE_EVENTS + _EVENT_HEADER_SIZE * self.num_events
    signal = utils.read_multiplexed_signal(
        self.path,
        lead_num=lead_num,
        total_leads=self.num_leads + self.num_events,
        start=start,
        end=end,
        samples_offset=offset,
        dtype=np_dtype
    )
    return signal * self._amplifier_range_in_volts / 2 ** self._conversion_bits
