"""Implementing the AbstractMultiLeadSignalReader interface for VHDR files."""

import configparser
import dataclasses
import functools
import os
from typing import Sequence

import more_itertools
import numpy as np

from dataset_creator.features.signal_io import base_signal_reader
from dataset_creator.features.signal_io import utils

# Specification for BrainVision's format can be found at:
# www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/
_HEADER_EXTENSION = '.vhdr'

# Sections in the header file:
_COMMON_INFOS = 'Common Infos'
_BINARY_INFOS = 'Binary Infos'
_CHANNEL_INFOS = 'Channel Infos'
_COMMENT_SECTION = '[Comment]'

# Keys of interest in the header file:
_DATA_FILE = 'DataFile'
_DATA_ORIENTATION = 'DataOrientation'
_DATA_POINTS = 'DataPoints'
_NUMBER_OF_CHANNELS = 'NumberOfChannels'
_SAMPLING_INTERVAL = 'SamplingInterval'
_BINARY_FORMAT = 'BinaryFormat'

# Allowed values
_VECTORIZED = 'VECTORIZED'
_MULTIPLEXED = 'MULTIPLEXED'

_FORMAT_TO_NP_DTYPE = {
    'IEEE_FLOAT_32': '<f4',
    'INT_16': '<i2',
    'INT_32': '<i4',
}

_MICRO_VOLT = base_signal_reader.PhysicalUnit.MICRO_VOLT


@dataclasses.dataclass
class BrainVisionLeadMetadata(base_signal_reader.LeadMetadata):
  resolution: float


class BrainVisionSignalReader(base_signal_reader.AbstractMultiLeadSignalReader):
  """An implementation of BrainVision's format for saving EEG files."""

  def __init__(self, path: str, **kwargs):
    basename, _ = os.path.splitext(path)
    header_path = basename + _HEADER_EXTENSION
    super().__init__(header_path, **kwargs)

  @functools.cached_property
  def num_leads(self) -> int:
    return int(self._config[_COMMON_INFOS][_NUMBER_OF_CHANNELS])

  @functools.cached_property
  def sampling_frequency(self) -> float:
    return 1e6 / float(self._config[_COMMON_INFOS][_SAMPLING_INTERVAL])

  @functools.cached_property
  def _config(self) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    # The vhdr file is not a valid INI file since its comments section can
    # include any free text, and the first line isn't a valid section / comment.
    # Remove these lines and parse the configuration from string.
    with open(self.path, 'r') as f:  # pylint: disable=unspecified-encoding
      lines = []
      for line in f:
        if _COMMENT_SECTION in line:
          break
        lines.append(line)
    config.read_string(''.join(lines[1:]))
    return config

  @functools.cached_property
  def lead_headers(self) -> Sequence[base_signal_reader.LeadMetadata]:
    headers = []
    for i, channel_info in enumerate(self._config[_CHANNEL_INFOS].values()):
      params = channel_info.split(',')
      label = params[0].replace('\1', ',')
      resolution = float(params[2]) if params[2] else 1.
      if len(params) > 3:
        unit = base_signal_reader.STRING_TO_PHYSICAL_UNIT.get(
            params[3], base_signal_reader.PhysicalUnit.UNKNOWN
        )
      else:
        unit = base_signal_reader.PhysicalUnit.MICRO_VOLT
      headers.append(
          BrainVisionLeadMetadata(
              lead_num=i, label=label, physical_dim=unit, resolution=resolution,
          )
      )
    return headers

  @property
  def total_samples(self) -> int:
    if _DATA_POINTS in self._config[_COMMON_INFOS]:
      return int(self._config[_COMMON_INFOS][_DATA_POINTS])
    data_stat = os.stat(self._data_path)
    return int(data_stat.st_size / (self.num_leads * self._sample_size))

  @property
  def _binary_format(self) -> str:
    return self._config[_BINARY_INFOS][_BINARY_FORMAT]

  @property
  def _sample_size(self) -> int:
    return np.dtype(_FORMAT_TO_NP_DTYPE[self._binary_format]).itemsize

  @functools.cached_property
  def _data_path(self) -> str:
    path = os.path.join(
        os.path.dirname(self.path),
        os.path.basename(self._config[_COMMON_INFOS][_DATA_FILE]),
    )
    if not os.path.exists(path) or not os.path.isfile(path):
      path = self.path.replace(_HEADER_EXTENSION, '.eeg')
    return path

  def _read_multiplexed_signal(
      self, lead_num: int, start: int, end: int
  ) -> np.ndarray:
    return utils.read_multiplexed_signal(
        self._data_path,
      lead_num,
      self.num_leads,
      start,
      end,
      dtype=_FORMAT_TO_NP_DTYPE[self._binary_format]
    )

  def _read_vectorized_signal(
      self, lead_num: int, start: int, end: int
  ) -> np.ndarray:
    offset = (lead_num * self.total_samples + start) * self._sample_size
    signal = np.fromfile(
        self._data_path,
        dtype=np.dtype(_FORMAT_TO_NP_DTYPE[self._binary_format]),
        offset=offset,
        count=end - start,
    )
    return signal.astype(np.float32)

  def _read_signal(self, lead_num: int, start: int, end: int) -> np.ndarray:
    orientation = self._config[_COMMON_INFOS][_DATA_ORIENTATION].upper()
    header = more_itertools.one(
        h for h in self.lead_headers if h.lead_num == lead_num
    )
    if orientation == _MULTIPLEXED:
      signal = self._read_multiplexed_signal(lead_num, start, end)
    else:
      assert orientation == _VECTORIZED
      signal = self._read_vectorized_signal(lead_num, start, end)
    return header.resolution * signal  # type: ignore[attr-defined]
