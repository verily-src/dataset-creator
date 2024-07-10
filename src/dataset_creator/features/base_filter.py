"""Implements the abstract Filter.

A Filter is a class which takes a signal and perform
low/band/high pass filtering, based on the given parameters.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Optional, Union

import numpy as np
import scipy.signal as scisig  # type: ignore[import]

from dataset_creator.features import serializable


class FilterType(enum.Enum):
  """An enumeration of the different types of filters."""
  LOW_PASS = "lowpass"
  BAND_PASS = "bandpass"
  HIGH_PASS = "highpass"

  def __str__(self) -> str:
    return self.value


class WindowType(enum.Enum):
  """An enumeration of the different types of windows to be used in filters.

  These values match the values accepted by scipy.signal.iirfilter.
  """
  BUTTER = "butter"
  CHEBY1 = "cheby1"
  CHEBY2 = "cheby2"
  ELLIP = "ellip"
  BESSEL = "bessel"

  def __str__(self) -> str:
    return self.value


@dataclasses.dataclass
class FilterConfig:
  """A configuration for a filter matching scipy.signal.iirfilter."""

  filter_type: FilterType
  cutoff: Union[float, tuple[float, float]]
  sampling_frequency: float  # Sampling frequency in [Hz]
  order: int = 4
  window_type: WindowType = WindowType.BUTTER
  ripple: Optional[float] = None
  min_attenuation: Optional[float] = None

  def __post_init__(self):
    """Validates that the parameters make a valid FilterConfig."""
    if self.sampling_frequency <= 0:
      raise ValueError("sampling_frequency must be > 0.")

    if self.filter_type in [FilterType.LOW_PASS, FilterType.HIGH_PASS] and not (
        isinstance(self.cutoff, (int, float))
    ):
      raise ValueError("When using LowPass/HighPass, cutoff must be scalar.")
    elif self.filter_type == FilterType.BAND_PASS and not (
        isinstance(self.cutoff, tuple) and len(self.cutoff) == 2
    ):
      raise ValueError(
          "When using BandPass, cutoff must be a sequence of length 2, where "
          "the first scalar marks the lowest frequency to pass, and the last "
          "scalar marks the highest frequency to pass."
      )

    if not isinstance(self.cutoff, tuple):
      frequencies = (self.cutoff,)
    else:
      frequencies = self.cutoff

    for frequency in frequencies:
      if frequency > self.sampling_frequency / 2:
        raise ValueError(
            "cutoff must smaller than the Nyquist frequency which is "
            f"{self.sampling_frequency / 2}."
        )
      if frequency < 0:
        raise ValueError("cutoff must be >= 0.")

  def __repr__(self) -> str:
    if isinstance(self.cutoff, tuple):
      cutoff_str = (
          f"{self.cutoff[0]},{self.cutoff[1]}"
      )
    else:
      cutoff_str = str(self.cutoff)

    return (
        f"{self.filter_type}|"
        f"{self.order}|"
        f"{cutoff_str}|"
        f"{self.ripple}|"
        f"{self.min_attenuation}|"
        f"{self.window_type}|"
        f"{self.sampling_frequency}"
    )


class BaseFilter(serializable.Serializable):
  """A Filter to process signals with.

  Filters are built to support spectral filtering of signals as part of our
  pre-processing pipeline. In addition, Filters are expected to be entirely
  reconstructible using the serialize and deserialize methods.
  """

  def __init__(self, filter_config: FilterConfig):
    """Initializes a Filter instance."""

    super().__init__()
    self._config = filter_config
    self._params = scisig.iirfilter(
        N=filter_config.order,
        Wn=filter_config.cutoff,
        rp=filter_config.ripple,
        rs=filter_config.min_attenuation,
        btype=filter_config.filter_type.value,
        analog=False,
        ftype=filter_config.window_type.value,
        output="sos",
        fs=filter_config.sampling_frequency,
    )

  def process(self, signal: np.ndarray) -> np.ndarray:
    """Spectrally filters the signal according to the filter's parameters.

    Args:
      signal: The signal to filter

    Returns:
      A NumPy array containing the filtered signal
    """

    return scisig.sosfiltfilt(self.params, signal)

  @property
  def params(self) -> np.ndarray:
    """Returns the filter's parameters.

    Returns:
      A NumPy array with the parameters of the filter, specified in a second
      order form (SOS) so as to be used with SciPy's sosfiltfilt method.
    """

    return self._params

  def serialize(self) -> bytes:
    return str(self._config).encode("utf-8")

  @classmethod
  def deserialize(cls, serialized: bytes) -> BaseFilter:
    decoded_config = serialized.decode("utf-8")
    (
        filter_type,
        order,
        cutoff,
        ripple,
        min_attenuation,
        window_type,
        fs,
    ) = decoded_config.split("|")

    if re.fullmatch(r"[0-9]+\.?[0-9]*", cutoff):
      cutoff_frequencies = float(cutoff)
    else:
      cutoff_frequencies = tuple(map(float, cutoff.split(",")))  # type: ignore

    attenuation = None if min_attenuation == "None" else float(min_attenuation)

    return cls(
        filter_config=FilterConfig(
            filter_type=FilterType(filter_type),
            order=int(order),
            cutoff=cutoff_frequencies,
            sampling_frequency=float(fs),
            window_type=WindowType(window_type),
            ripple=float(ripple) if ripple != "None" else None,
            min_attenuation=attenuation,
        )
    )
