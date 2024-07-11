"""Implementing the AbstractMultiLeadSignalReader interface for SET files."""

import dataclasses
import functools
import os
from typing import Any, Mapping, Optional, Sequence, Union

import h5py  # type: ignore[import]
import numpy as np
import scipy.io  # type: ignore[import]

from dataset_creator.features.signal_io import base_signal_reader
from dataset_creator.features.signal_io import utils

# Specification for EEGLAB's format can be found at:
# https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html
_SET_EXTENSION = '.set'
_FDT_EXTENSION = '.fdt'

_NUM_CHANNELS_KEY = 'nbchan'
_SAMPLING_RATE_KEY = 'srate'
_CHANNEL_LABELS_KEY = 'chanlocs'
_NUM_SAMPLES_KEY = 'pnts'
_DATA_PATH_KEY = 'datfile'
_TRIALS_KEY = 'trials'
_EEG_DATASET_KEY = 'EEG'
_DATA_KEY = 'data'
_ALLEEG_DATASET_KEY = 'ALLEEG'

_LABELS_KEY = 'labels'

_MICRO_VOLT = base_signal_reader.PhysicalUnit.MICRO_VOLT


def _array_to_str(np_array: np.ndarray) -> str:
  return bytes(np_array.astype('uint8')).decode().rstrip('\x00')


def _slice_h5_dataset(
    dataset: h5py.Dataset,
    slice_dims: Sequence[int],
    slice_indexes: Sequence[int]
) -> np.ndarray:
  num_dims = len(dataset.shape)
  sliced: list[Union[slice, int]] = [slice(None) for _ in range(num_dims)]
  for dim, index in zip(slice_dims, slice_indexes):
    sliced[dim] = index
  return dataset[tuple(sliced)]


def _h5_squeeze(h5_object: h5py.Dataset) -> np.ndarray:
  squeeze_dims = [dim for dim, size in enumerate(h5_object.shape) if size == 1]
  return _slice_h5_dataset(h5_object, squeeze_dims, [0] * len(squeeze_dims))


def _load_h5(path: str) -> Mapping[str, Any]:
  f = h5py.File(path)
  return f[_EEG_DATASET_KEY] if _EEG_DATASET_KEY in f else f


@dataclasses.dataclass(frozen=True)
class _SetHeader:
  num_channels: int
  sampling_rate: float
  num_samples: int
  data_path: str


class EEGLABSignalReader(base_signal_reader.AbstractMultiLeadSignalReader):
  """An implementation of EEGLAB's format for saving EEG files."""

  def __init__(self, path: str, **kwargs):
    basename, _ = os.path.splitext(path)
    header_path = basename + _SET_EXTENSION
    super().__init__(header_path, **kwargs)

  @functools.cached_property
  def num_leads(self) -> int:
    return self._config.num_channels

  @functools.cached_property
  def sampling_frequency(self) -> float:
    return self._config.sampling_rate

  def _load_variables(self, variable_names: Sequence[str]) -> Mapping[str, Any]:
    if not h5py.is_hdf5(self.path):
      variables = list(variable_names) + [_EEG_DATASET_KEY, _ALLEEG_DATASET_KEY]
      loaded = scipy.io.loadmat(
          self.path,
          variable_names=variables,
          simplify_cells=True,
          squeeze_me=True
      )
      if _EEG_DATASET_KEY in loaded or _ALLEEG_DATASET_KEY in loaded:
        loaded = loaded.get(_EEG_DATASET_KEY, loaded.get(_ALLEEG_DATASET_KEY))
      return {
          k: v for k, v in loaded.items()
          if not isinstance(v, np.ndarray) or v.size > 0
      }
    f = _load_h5(self.path)
    return {name: _h5_squeeze(f[name]) for name in variable_names if name in f}

  @functools.cached_property
  def _config(self) -> _SetHeader:
    variable_names = [
        _NUM_CHANNELS_KEY, _SAMPLING_RATE_KEY, _NUM_SAMPLES_KEY, _DATA_PATH_KEY,
    ]
    loaded = self._load_variables(variable_names)
    data_path = loaded.get(_DATA_PATH_KEY)
    if h5py.is_hdf5(self.path):
      assert isinstance(data_path, np.ndarray)
      data_path = _array_to_str(data_path)
    if not data_path or not os.path.isfile(data_path):
      fdt_potential_path = self.path.replace(_SET_EXTENSION, _FDT_EXTENSION)
      if os.path.isfile(fdt_potential_path):
        data_path = fdt_potential_path
      else:
        data_path = self.path
    return _SetHeader(
        num_channels=loaded.get(_NUM_CHANNELS_KEY, 0),
        sampling_rate=loaded.get(_SAMPLING_RATE_KEY, 0.),
        num_samples=loaded.get(_NUM_SAMPLES_KEY, 0),
        data_path=data_path,
    )

  @functools.cached_property
  def lead_headers(self) -> Sequence[base_signal_reader.LeadMetadata]:
    loaded = self._load_variables([_CHANNEL_LABELS_KEY])
    assert _CHANNEL_LABELS_KEY in loaded
    channel_locations = loaded[_CHANNEL_LABELS_KEY]
    labels = [location[_LABELS_KEY] for location in channel_locations]

    headers = []
    for i, label in enumerate(labels):
      label = _array_to_str(label) if isinstance(label, np.ndarray) else label
      headers.append(base_signal_reader.LeadMetadata(
          lead_num=i, label=label, physical_dim=_MICRO_VOLT
      ))
    return headers

  @property
  def total_samples(self) -> int:
    return self._config.num_samples

  def _read_signal(
      self, lead_num: int, start: int, end: int
  ) -> Optional[np.ndarray]:
    data_path = os.path.join(
        os.path.dirname(self.path), os.path.basename(self._config.data_path)
    )
    if data_path.endswith(_SET_EXTENSION):
      assert os.path.basename(self.path) == os.path.basename(data_path)
      if not h5py.is_hdf5(data_path):
        loaded = self._load_variables([_DATA_KEY])
        assert _DATA_KEY in loaded
        return loaded[_DATA_KEY][lead_num][start:end]
      f = _load_h5(data_path)
      if _h5_squeeze(f[_TRIALS_KEY]) <= 1:
        dims = [1]
        indices = [lead_num]
      else:
        dims = [0, 2]
        indices = [0, lead_num]
      return _slice_h5_dataset(f[_DATA_KEY], dims, indices)[start:end]

    return utils.read_multiplexed_signal(
        data_path, lead_num, self.num_leads, start, end
    )
