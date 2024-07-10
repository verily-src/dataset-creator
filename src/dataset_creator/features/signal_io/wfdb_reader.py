"""Reader implementation for wfdb supported formats."""
from __future__ import annotations

import dataclasses
import functools
import os
from typing import Union

import numpy as np
import wfdb  # type: ignore[import]

from dataset_creator.features.signal_io import base_signal_reader


@dataclasses.dataclass(frozen=True)
class _WfdbMetadata:
  num_channels: int
  sampling_rate: float
  num_samples: int


class WfdbReader(base_signal_reader.AbstractMultiLeadSignalReader):
  """Reader implementation for wfdb supported formats."""

  def __init__(self, path: str, **kwargs):
    super().__init__(path, **kwargs)
    self.basename, _ = os.path.splitext(self.path)

  @functools.cached_property
  def num_leads(self) -> int:
    return self._config.num_channels

  @functools.cached_property
  def sampling_frequency(self) -> float:
    """Returns the sampling frequency of the signal."""
    return self._config.sampling_rate

  @property
  def total_samples(self) -> int:
    return self._config.num_samples

  @functools.cached_property
  def _load_header(
      self,
  ) -> Union[wfdb.io.record.MultiRecord, wfdb.io.record.Record]:
    """Loads dicom dataset from gcs."""
    return wfdb.rdheader(self.basename)

  @functools.cached_property
  def _config(self) -> _WfdbMetadata:
    """Extracts metadata from self.path."""
    metadata = self._load_header
    return _WfdbMetadata(
        num_channels=metadata.n_sig,
        sampling_rate=metadata.fs,
        num_samples=metadata.sig_len,
    )

  @functools.cached_property
  def lead_headers(self) -> list[base_signal_reader.LeadMetadata]:
    """Extracts leads metadata from self.path."""
    metadata = self._load_header
    headers = []
    for i, (label, units) in enumerate(zip(metadata.sig_name, metadata.units)):
      physical_dim = base_signal_reader.STRING_TO_PHYSICAL_UNIT[units]
      headers.append(
          base_signal_reader.LeadMetadata(
              lead_num=i, label=label, physical_dim=physical_dim
          )
      )
    return headers

  def _read_signal(
      self, lead_num: int, start: int, end: int
  ) -> np.ndarray:
    """Reads the signal from the given channel."""
    signal = wfdb.rdrecord(
        self.basename, channels=[lead_num], physical=False, sampfrom=start,
        sampto=end).d_signal.flatten()
    return signal
