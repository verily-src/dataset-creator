"""A module for reading average-montaged Multi-lead signals."""

import functools
import os
import re
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import numpy.typing  # pylint: disable=unused-import

from dataset_creator.features import base_feature
from dataset_creator.features import base_filter
from dataset_creator.features import fields
from dataset_creator.features.signal_io import base_signal_reader
from dataset_creator.features.signal_io import signal_reader

_BRAIN_VISION_EXTENSIONS = ['.vhdr', '.eeg', '.avg', '.seg']
_CONTINUOUS_EXTENSIONS = ['.cnt']
_EDF_READER_EXTENSIONS = ['.edf', '.bdf']
_EEGLAB_EXTENSIONS = ['.set', '.fdt']
_SIMPLE_BINARY_EXTENSIONS = ['.raw']
_WFDB_EXTENSIONS = ['.hea']
_MFF_EXTENSIONS = ['.mff']

_REFERENCE_INDEX = -1
_REFERENCE_LEAD_NUM = -1
_EMPTY_INDEX = -2
_EMPTY_LEAD_NUM = -2



def get_default_reader(
    path: str, **kwargs
) -> base_signal_reader.AbstractMultiLeadSignalReader:
  """Returns the standard SignalReader for this path.

  Args:
    path: Path of the signal to be read.
    **kwargs: Additional kwargs to pass to the reader instantiation.
  """
  _, extension = os.path.splitext(path.rstrip('/'))
  extension = extension.lower()
  if extension in _BRAIN_VISION_EXTENSIONS:
    return signal_reader.BrainVisionSignalReader(path, **kwargs)
  if extension in _CONTINUOUS_EXTENSIONS:
    return signal_reader.ContinuousSignalReader(path, **kwargs)
  if extension in _EDF_READER_EXTENSIONS:
    return signal_reader.EdfSignalReader(path, **kwargs)
  if extension in _EEGLAB_EXTENSIONS:
    return signal_reader.EEGLABSignalReader(path, **kwargs)
  if extension in _SIMPLE_BINARY_EXTENSIONS:
    return signal_reader.SimpleBinarySignalReader(path, **kwargs)
  if extension in _WFDB_EXTENSIONS:
    return signal_reader.WfdbReader(path, **kwargs)
  if extension in _MFF_EXTENSIONS:
    return signal_reader.MneSignalReader(path, **kwargs)
  raise NotImplementedError(f'Unsupported extension: {extension}')


def _resample(signal: np.ndarray, sampling_ratio: float) -> np.ndarray:
  indices = np.arange(0, len(signal), sampling_ratio)
  upper_indices = np.minimum(np.ceil(indices).astype(np.int32), len(signal) - 1)
  lower_indices = indices.astype(np.int32)
  # Defines which part of the ceiled index is taken into account:
  parts = np.remainder(indices, 1)
  return signal[upper_indices] * parts + signal[lower_indices] * (1 - parts)


class MultiLeadSignalFeature(base_feature.CustomFeature):
  """A feature that converts a video reader and timestamps to frames."""

  _serializable_classes = (
      signal_reader.BrainVisionSignalReader,
      signal_reader.ContinuousSignalReader,
      signal_reader.EdfSignalReader,
      signal_reader.EEGLABSignalReader,
      signal_reader.MneSignalReader,
      signal_reader.SimpleBinarySignalReader,
      signal_reader.WfdbReader,
      base_filter.BaseFilter,
  )

  def __init__(
      self,
      reader: base_signal_reader.AbstractMultiLeadSignalReader,
      label_patterns: Optional[Sequence[str]] = None,
      start: int = 0,
      end: int = -1,
      resample_at: Optional[float] = None,
      reference_lead: str = '',
      signal_filter: Optional[base_filter.BaseFilter] = None,
      empty_leads: Optional[str] = None,
      **kwargs
  ):
    """Instantiate an MultiLeadSignalFeature.

    Args:
      reader: An AbstractMultiLeadSignalReader used to read the signal.
      label_patterns: A sequence of regex patterns of leads to include. Default
        is to include all leads.
      start: The 0-based sample number to start the read operation from
        (including). Please note that this sample number refers to the signal
        in its original sampling rate, before resampling.
      end: The 0-based last sample number to read (excluding). Please note that
        this sample number refers to the signal in its original sampling rate,
        before resampling.
      resample_at: Downsample the signal to this frequency. If None, the
        original sampling frequency is preserved.
      reference_lead: The name of the lead used as reference while recording.
      signal_filter: The filter to apply to the signal after reading it.
      empty_leads: The name (label) of empty leads. If None, empty leads are not
        provided. Default: None.
      **kwargs: Additional keyword arguments to be passed to CustomFeature.
    """
    if resample_at is not None:
      resample_at = float(resample_at)  # Force float even if an int is given
    super().__init__(
        reader=reader,
        label_patterns=label_patterns,
        start=start,
        end=end,
        resample_at=resample_at,
        reference_lead=reference_lead,
        signal_filter=signal_filter,
        empty_leads=empty_leads,
        **kwargs
    )
    self._reader = reader
    self._label_patterns = label_patterns
    self._start = start
    self._end = end
    self._filter = signal_filter
    self._resample_at = resample_at
    self._reference_lead = reference_lead
    self._empty_leads = empty_leads

  @functools.cached_property
  def _relevant_header_indices(self) -> Sequence[int]:
    """Returns the indices of headers matching self._label_patterns."""
    if not self._label_patterns:
      return [header.lead_num for header in self._reader.lead_headers]
    indices = []
    for pattern in self._label_patterns:
      matching_indices = [
          i for i, header in enumerate(self._reader.lead_headers)
          if re.search(pattern, header.label)
      ]
      if self._reference_lead and re.search(pattern, self._reference_lead):
        matching_indices.append(_REFERENCE_INDEX)
      if len(matching_indices) > 1:
        raise RuntimeError(
            f'{pattern} matches several labels in {self._reader.path}.'
        )
      elif not matching_indices:
        if self._empty_leads:
          indices.append(_EMPTY_INDEX)
        else:
          raise RuntimeError(
            f'{pattern} matches no labels in {self._reader.path}. If you desire'
            ' to fill empty leads in such cases, please use the empty_lead'
            ' argument.'
        )
      else:
        indices.append(matching_indices[0])
    return indices

  def split(self) -> list[int]:
    """See base class."""
    try:
      headers = self._reader.lead_headers
    except FileNotFoundError:
      # In case of an invalid file.
      return []

    try:
      indices = self._relevant_header_indices
    except RuntimeError:
      return []

    lead_nums = []
    for i in indices:
      if i == _REFERENCE_INDEX:
        lead_nums.append(_REFERENCE_LEAD_NUM)
      elif i == _EMPTY_INDEX:
        lead_nums.append(_EMPTY_LEAD_NUM)
      else:
        lead_nums.append(headers[i].lead_num)
    return lead_nums


  def process(self, metadata_value: int, _: Any) -> Optional[np.ndarray]:
    """Reads the signal from the given lead using self._reader.

    Args:
      metadata_value: The number of channel to read.

    Returns:
      A np.ndarray corresponding to the lead number and self.reader, or None in
      case the read operation failed.
    """

    if metadata_value == _REFERENCE_LEAD_NUM: # Reference lead
      end = self._reader.total_samples if self._end == -1 else self._end
      signal = np.zeros(end - self._start)
    elif metadata_value == _EMPTY_LEAD_NUM: # Empty lead
      end = self._reader.total_samples if self._end == -1 else self._end
      signal = - np.ones(end - self._start)
    else:
      signal = self._reader.read_signal(metadata_value, self._start, self._end)
      if signal is None:
        return None
    if self._filter is not None and metadata_value != _EMPTY_LEAD_NUM:
      signal = self._filter.process(signal)
    if self._resample_at is None:
      return signal
    sampling_ratio = self._reader.sampling_frequency / self._resample_at
    return _resample(signal, sampling_ratio)

  def merge(
      self, values: Sequence[np.ndarray]
  ) -> dict[str, base_feature.ValueFeature]:
    """Merges the read images and encodes them.

    Args:
      values: The sequence of read signals. The expected shape is
        (num_leads, samples_per_lead).

    Returns:
      A dictionary containing the following keys:
        fields.EDF_CHANNEL_LABELS: The list of labels of the different channels.
        fields.EDF_SIGNAL: A np.ndarray containing the different signal channels
          stacked in the same array. The order of signals match the labels under
          fields.EDF_CHANNEL_LABELS.
    """
    if not values:
      return {}
    headers = self._reader.lead_headers
    labels = []
    for i in self._relevant_header_indices:
      if i == _REFERENCE_INDEX:
        labels.append(self._reference_lead)
      elif i == _EMPTY_INDEX:
        assert self._empty_leads is not None
        labels.append(self._empty_leads)
      else:
        labels.append(headers[i].label)
    physical_dims = []
    for i in self._relevant_header_indices:
      if i == _REFERENCE_INDEX:
        physical_dims.append(base_signal_reader.PhysicalUnit.MICRO_VOLT.value)
      elif i == _EMPTY_INDEX:
        physical_dims.append(base_signal_reader.PhysicalUnit.UNKNOWN.value)
      else:
        physical_dims.append(headers[i].physical_dim.value)
    frequency = self._resample_at or self._reader.sampling_frequency

    multi_lead_signal = np.stack(values).astype(np.float32)
    non_empty_indices = [i for i, label in enumerate(labels)
                         if label != self._empty_leads]
    # Ensuring the reference of each lead is always the average of all leads.
    multi_lead_signal[non_empty_indices] -= (
        np.mean(multi_lead_signal[non_empty_indices], axis=0, keepdims=True)
    )
    return {
        fields.MULTI_LEAD_LABELS: labels,
        fields.MULTI_LEAD_START_SAMPLE: self._start,
        fields.MULTI_LEAD_END_SAMPLE: self._end,
        fields.MULTI_LEAD_FREQUENCY: frequency,
        fields.MULTI_LEAD_PHYSICAL_UNITS: physical_dims,
        fields.MULTI_LEAD_SIGNAL: multi_lead_signal,
        fields.MULTI_LEAD_PATH: self._reader.path
    }

  @property
  def output_signature(self) -> Mapping[str, Any]:
    return {
        fields.MULTI_LEAD_LABELS: list[str],
        fields.MULTI_LEAD_START_SAMPLE: int,
        fields.MULTI_LEAD_END_SAMPLE: int,
        fields.MULTI_LEAD_FREQUENCY: float,
        fields.MULTI_LEAD_PHYSICAL_UNITS: list[float],
        fields.MULTI_LEAD_SIGNAL: np.typing.NDArray[np.float32],
        fields.MULTI_LEAD_PATH: str,
    }
