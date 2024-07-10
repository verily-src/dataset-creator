"""A module implementing an AbstractMultiLeadSignalReader for .cnt files."""

import dataclasses
import functools
import struct
from typing import Sequence

import more_itertools
import numpy as np

from dataset_creator.features.signal_io import base_signal_reader

# Specification for the continuous format can be found at:
# https://paulbourke.net/dataformats/eeg/
SETUP_HEADER_SIZE = 900
_CHANNEL_INFO_SIZE = 75

_MAX_CHANNEL_NAME_LENGTH = 10

_BASELINE_OFFSET = 47
_SENSITIVITY_OFFSET = 59
_CALIBRATION_OFFSET = 71
_NUM_CHANNELS_OFFSET = 370
_SAMPLING_FREQUENCY_OFFSET = 376
_TOTAL_SAMPLES_OFFSET = 864
_EVENTS_OFFSET_OFFSET = 886
_CHANNEL_OFFSET_OFFSET = 894

_D2A_SCALING = 204.8

_ALLOWED_CONTINUOUS_EXTENSIONS = ['.cnt']


def _read_from_path(path: str, offset: int, fmt: str) -> int:
  with open(path, 'rb') as f:
    f.seek(offset)
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


@dataclasses.dataclass
class ContinuousLeadMetadata(base_signal_reader.LeadMetadata):
  baseline: float
  micro_volt_factor: float


class ContinuousSignalReader(base_signal_reader.AbstractMultiLeadSignalReader):
  """A reader for .cnt files."""

  def __init__(self, path: str, **kwargs):
    kwargs['allowed_extensions'] = _ALLOWED_CONTINUOUS_EXTENSIONS
    super().__init__(path, **kwargs)

  @functools.cached_property
  def num_leads(self) -> int:
    """See base class."""
    return _read_from_path(self.path, _NUM_CHANNELS_OFFSET, fmt='<H')

  @functools.cached_property
  def sampling_frequency(self) -> float:
    """See base class."""
    return _read_from_path(self.path, _SAMPLING_FREQUENCY_OFFSET, fmt='<H')

  @functools.cached_property
  def total_samples(self) -> int:
    """See base class."""
    events_offset = _read_from_path(self.path, _EVENTS_OFFSET_OFFSET, '<I')
    total_header_size = SETUP_HEADER_SIZE + _CHANNEL_INFO_SIZE * self.num_leads
    total_records_size = events_offset - total_header_size
    self._sample_size = 2
    return int(total_records_size / (self._sample_size * self.num_leads))

  @functools.cached_property
  def _num_samples_per_block_per_lead(self) -> int:
    channel_offset = _read_from_path(self.path, _CHANNEL_OFFSET_OFFSET, '<I')
    return channel_offset if channel_offset > 0 else 1

  @functools.cached_property
  def lead_headers(self) -> Sequence[ContinuousLeadMetadata]:
    headers = []
    with open(self.path, 'rb') as f:
      for i in range(self.num_leads):
        f.seek(SETUP_HEADER_SIZE + i * _CHANNEL_INFO_SIZE)
        channel_name_buffer = f.read(_MAX_CHANNEL_NAME_LENGTH)
        null_character_index = channel_name_buffer.find(0)
        if null_character_index != -1:
          channel_name_buffer = channel_name_buffer[:null_character_index]
        channel_name = channel_name_buffer.decode()

        f.seek(SETUP_HEADER_SIZE + i * _CHANNEL_INFO_SIZE + _BASELINE_OFFSET)
        baseline = struct.unpack('<h', f.read(2))[0]
        f.seek(SETUP_HEADER_SIZE + i * _CHANNEL_INFO_SIZE + _SENSITIVITY_OFFSET)
        sensitivity = struct.unpack('<f', f.read(4))[0]
        f.seek(SETUP_HEADER_SIZE + i * _CHANNEL_INFO_SIZE + _CALIBRATION_OFFSET)
        calibration = struct.unpack('<f', f.read(4))[0]

        headers.append(
            ContinuousLeadMetadata(
                lead_num=i,
                label=channel_name,
                physical_dim=base_signal_reader.PhysicalUnit.MICRO_VOLT,
                baseline=baseline,
                micro_volt_factor=calibration * sensitivity / _D2A_SCALING,
            )
        )
    return headers

  def _read_signal(self, lead_num: int, start: int, end: int) -> np.ndarray:
    # Access total_samples member so the _sample_size attribute is initialized.
    assert self.total_samples >= end - start

    headers_end_offset = SETUP_HEADER_SIZE + _CHANNEL_INFO_SIZE * self.num_leads
    samples_per_block_per_lead = self._num_samples_per_block_per_lead
    samples_per_block = samples_per_block_per_lead * self.num_leads
    block_size = samples_per_block * self._sample_size
    start_block_num = int(start // samples_per_block_per_lead)
    start_block_offset = headers_end_offset + start_block_num * block_size
    last_block_num = int((end - 1) // samples_per_block_per_lead)

    header = more_itertools.one(
        [header for header in self.lead_headers if header.lead_num == lead_num]
    )
    num_blocks_to_read = last_block_num - start_block_num + 1
    np_dtype = f'<i{self._sample_size}'
    all_channels_signal = np.fromfile(
        self.path,
        dtype=np_dtype,
        count=num_blocks_to_read * samples_per_block,
        offset=start_block_offset,
    ).reshape((num_blocks_to_read, self.num_leads, samples_per_block_per_lead))
    signal = all_channels_signal[:, lead_num, :].flatten()
    signal = (signal - header.baseline) * header.micro_volt_factor

    start_index = start % samples_per_block_per_lead
    return signal[start_index: start_index + (end - start)]
