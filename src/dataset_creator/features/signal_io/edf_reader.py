"""Implementing the AbstractMultiLeadSignalReader interface for EDF files."""

import dataclasses
import functools
import os
import struct
from typing import BinaryIO, Iterable

import more_itertools
import numpy as np

from dataset_creator.features.signal_io import base_signal_reader

# These constants are derived from the EDF specification:
# https://www.edfplus.info/specs/edf.html
_RECORDS_OFFSET_OFFSET = 0xB8
_NUM_RECORDS_OFFSET = 0xEC
_RECORD_DURATION_OFFSET = 0xF4
_NUM_SIGNALS_OFFSET = 0xFC
_SIGNAL_HEADERS_OFFSET = 0x100

_PHYSICAL_DIM_SIZE = 8
_DEFAULT_FLOAT_SIZE = 8
_DEFAULT_BUFFER_SIZE = 80
_INT_SIZE_FOR_NUM_SIGNALS = 4
_LABEL_LENGTH = 16

_EDF_ANNOTATIONS = 'EDF Annotations'


def _float_bytes_to_float(int_bytes: bytes) -> float:
  return float(int_bytes.decode().rstrip(' '))


@dataclasses.dataclass
class _EdfLeadMetadata(base_signal_reader.LeadMetadata):
  """A helper class containing the metadata for a specific signal."""
  physical_min: float
  physical_max: float
  digital_min: float
  digital_max: float
  samples_per_record: int

  def __post_init__(self):
    # Calculate the parameters for transferring from the digital values to the
    # physical values using y = m * (x + b).
    digital_diff = self.digital_max - self.digital_min
    physical_diff = self.physical_max - self.physical_min
    self.m = 1
    self.b = 0
    if digital_diff and physical_diff:
      self.m = physical_diff / digital_diff
      self.b = self.physical_max / self.m - self.digital_max


class EdfSignalReader(base_signal_reader.AbstractMultiLeadSignalReader):
  """An EDF reader which supports CNS paths, and can be reused."""

  def __init__(self, *args, **kwargs):
    kwargs['allowed_extensions'] = ['.edf', '.bdf']
    super().__init__(*args, **kwargs)
    ext_to_sample_size = {'.edf': 2, '.bdf': 3}
    _, ext = os.path.splitext(self.path)
    self._sample_size = ext_to_sample_size[ext]

  def _read_float_list(
      self, f: BinaryIO, size: int = _DEFAULT_FLOAT_SIZE
  ) -> list[float]:
    return [_float_bytes_to_float(f.read(size)) for _ in range(self.num_leads)]

  def _read_str_list(
      self, f: BinaryIO, size: int = _LABEL_LENGTH
  ) -> list[str]:
    return [f.read(size).decode().rstrip(' ') for _ in range(self.num_leads)]

  @functools.cached_property
  def num_leads(self) -> int:
    """See base class."""
    with open(self.path, 'rb') as f:
      f.seek(_NUM_SIGNALS_OFFSET)
      return int(_float_bytes_to_float(f.read(_INT_SIZE_FOR_NUM_SIGNALS)))

  @functools.cached_property
  def _num_records(self) -> int:
    with open(self.path, 'rb') as f:
      f.seek(_NUM_RECORDS_OFFSET)
      return int(_float_bytes_to_float(f.read(_DEFAULT_FLOAT_SIZE)))

  @functools.cached_property
  def _records_offset(self) -> int:
    with open(self.path, 'rb') as f:
      f.seek(_RECORDS_OFFSET_OFFSET)
      return int(_float_bytes_to_float(f.read(_DEFAULT_FLOAT_SIZE)))

  @functools.cached_property
  def sampling_frequency(self) -> float:
    """See base class."""
    with open(self.path, 'rb') as f:
      f.seek(_RECORD_DURATION_OFFSET)
      record_duration = _float_bytes_to_float(f.read(_DEFAULT_FLOAT_SIZE))
    # Take the samples_per_record of any channel outside _EDF_ANNOTATIONS
    samples_per_record = more_itertools.first(
        (
            header.samples_per_record for header in self.lead_headers
            if header.label != _EDF_ANNOTATIONS
        )
    )
    return samples_per_record / record_duration

  @functools.cached_property
  def lead_headers(self) -> list[_EdfLeadMetadata]:
    """See base class."""
    with open(self.path, 'rb') as f:
      f.seek(_SIGNAL_HEADERS_OFFSET)
      labels = self._read_str_list(f)

      f.seek(_DEFAULT_BUFFER_SIZE * self.num_leads, os.SEEK_CUR)
      physical_dims = self._read_str_list(f, size=_PHYSICAL_DIM_SIZE)
      physical_mins = self._read_float_list(f)
      physical_maxs = self._read_float_list(f)
      digital_mins = self._read_float_list(f)
      digital_maxs = self._read_float_list(f)

      f.seek(_DEFAULT_BUFFER_SIZE * self.num_leads, os.SEEK_CUR)
      samples_per_record = self._read_float_list(f)

    headers = []
    for i in range(self.num_leads):
      unit = base_signal_reader.STRING_TO_PHYSICAL_UNIT.get(
          physical_dims[i], base_signal_reader.PhysicalUnit.UNKNOWN
      )
      headers.append(
          _EdfLeadMetadata(
              lead_num=i,
              label=labels[i],
              physical_dim=unit,
              physical_min=physical_mins[i],
              physical_max=physical_maxs[i],
              digital_min=digital_mins[i],
              digital_max=digital_maxs[i],
              samples_per_record=int(samples_per_record[i]),
          )
      )

    # Sort the headers to be in deterministic order unrelated of the order in
    # the EDF file.
    headers.sort(key=lambda header: header.label)
    return headers

  @property
  def total_samples(self) -> int:
    """See base class."""
    total_samples_per_channel = [
        header.samples_per_record * self._num_records
        for header in self.lead_headers
        if header.label != _EDF_ANNOTATIONS
    ]
    return more_itertools.one(set(total_samples_per_channel))

  def _parse_record_bytes(self, record_bytes: bytes) -> Iterable[int]:
    num_entries = int(len(record_bytes) / self._sample_size)
    if self._sample_size == 2:
      return struct.unpack('<' + 'h' * num_entries, record_bytes)
    return (
        int.from_bytes(int_bytes, byteorder='little', signed=True)
        for int_bytes in more_itertools.chunked(record_bytes, self._sample_size)
    )

  def _get_num_samples_per_record(
      self, samples_per_record: int, start: int, end: int, record_num: int,
  ) -> int:
    first_record = int(start / samples_per_record)
    last_record = int((end - 1) / samples_per_record)

    if record_num == first_record:
      if first_record == last_record:
        return end - start
      return samples_per_record - start % samples_per_record
    elif first_record < record_num < last_record:
      return samples_per_record
    assert record_num == last_record
    return (end - 1) % samples_per_record + 1

  def _read_signal(self, lead_num: int, start: int, end: int) -> np.ndarray:
    """Reads the signal from the given channel from self.path.

    Args:
      lead_num: The lead number to read.
      start: The 0-based sample number to start at.
      end: The 0-based end sample to end at (exclusive). If <0, reads until the
        end of the signal. Default is -1.

    Returns:
      A np.ndarray with shape (total_samples,) if lead_num is valid, else None.
    """
    header = more_itertools.one(
        (h for h in self.lead_headers if h.lead_num == lead_num)
    )
    samples_per_record = header.samples_per_record

    record_size = sum(h.samples_per_record for h in self.lead_headers)
    record_size *= self._sample_size

    first_record = int(start / samples_per_record)
    last_record = int((end - 1) / samples_per_record)

    first_record_offset = first_record * record_size
    lead_offset = sum(
        h.samples_per_record * self._sample_size
        for h in self.lead_headers
        if h.lead_num < lead_num
    )
    start_offset = self._records_offset + first_record_offset + lead_offset
    start_offset += (start % samples_per_record) * self._sample_size

    with open(self.path, 'rb') as f:
      # Explicitly read the first record, as it might not start at the start of
      # the record.
      f.seek(start_offset)
      to_read = self._get_num_samples_per_record(
          samples_per_record, start, end, first_record
      )
      to_read *= self._sample_size
      values = list(self._parse_record_bytes(f.read(to_read)))
      # Seek to the start of lead_num in the next record:
      f.seek(record_size - samples_per_record * self._sample_size, os.SEEK_CUR)
      for record_num in range(first_record + 1, last_record + 1):
        to_read = self._get_num_samples_per_record(
            samples_per_record, start, end, record_num
        )
        assert to_read
        to_read *= self._sample_size
        buffer = f.read(to_read)
        values.extend(self._parse_record_bytes(buffer))
        f.seek(record_size - to_read, os.SEEK_CUR)

    return np.array(
        [header.m * (x + header.b) for x in values], dtype=np.float32
    )
