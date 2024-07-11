"""Implements a reader for the MNE-supported file formats."""

import functools
from typing import Optional, Sequence

import mne  # type: ignore[import]
import numpy as np

from dataset_creator.features.signal_io import base_signal_reader

_SAMPLING_FREQ_ENTRY = 'sfreq'


class MneSignalReader(base_signal_reader.AbstractMultiLeadSignalReader):
  """A reader for MNE-supported file formats."""

  def __init__(self, path: str, **kwargs):
    super().__init__(path, **kwargs)
    if self._check_path:
      _ = self._mne_raw

  @functools.cached_property
  def _mne_raw(self) -> mne.io.Raw:
    return mne.io.read_raw(self.path, preload=False, verbose=False)

  @functools.cached_property
  def num_leads(self) -> int:
    """See base class."""
    return len(self.lead_headers)

  @functools.cached_property
  def lead_headers(self) -> Sequence[base_signal_reader.LeadMetadata]:
    """See base class."""
    return [
        base_signal_reader.LeadMetadata(
            lead_num=i,
            label=label,
            physical_dim=base_signal_reader.PhysicalUnit.MICRO_VOLT,
        ) for i, label in enumerate(self._mne_raw.ch_names)
    ]

  @functools.cached_property
  def sampling_frequency(self) -> float:
    """See base class."""
    return float(self._mne_raw.info[_SAMPLING_FREQ_ENTRY])

  @property
  def total_samples(self) -> int:
    """See base class."""
    return int(self._mne_raw.n_times)

  def _read_signal(
      self, lead_num: int, start: int, end: int
  ) -> Optional[np.ndarray]:
    """See base class."""
    signal = self._mne_raw.get_data(picks=[lead_num], start=start, stop=end)
    return signal[0] * 1e6
