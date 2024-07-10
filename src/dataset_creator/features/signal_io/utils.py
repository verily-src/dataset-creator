"""Utilities to be used by signal readers."""

import numpy as np


def read_multiplexed_signal(
    path: str,
    lead_num: int,
    total_leads: int,
    start: int,
    end: int,
    samples_offset: int = 0,
    dtype: str = '<f4',
) -> np.ndarray:
  """Reads a multiplexed signal into a numpy array.

  Args:
    path: The path to the multiplexed file,
    lead_num: The signal channel we wish to read.
    total_leads: The total number of leads in this file.
    start: The first sample (inclusive) to read.
    end: The last sample (exclusive) to read.
    samples_offset: An offset the first sample starts in.
    dtype: The dtype of each sample in the file.

  Returns:
    A np.ndarray with shape (total_samples_in_lead,).
  """
  multi_signal = np.fromfile(
      path,
      dtype=dtype,
      count=total_leads * (end - start),
      offset=samples_offset + start * total_leads * np.dtype(dtype).itemsize
  )
  return multi_signal[lead_num::total_leads]
