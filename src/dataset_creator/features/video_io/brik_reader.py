"""A module implementing a BRIK reader."""

import functools
from typing import Callable

import nibabel as nib

from dataset_creator.features.video_io import nibabel_reader

_UNIT_MSEC = 77001
_UNIT_SEC = 77002
_UNIT_HZ = 77003


class BrikReader(nibabel_reader.NibabelReader):
  """A reader for .HEAD and .BRIK AFNI files.

  Attributes:
    video_path: The path of the file this reader reads from.

  Raises:
    FileNotFoundError: If check_path and the file does not exist.
    ImageFileError: If check_path and the file is not a valid AFNI file.
    IndexError: If the z_slice is invalid.
  """

  @functools.cached_property
  def fps(self) -> float:
    """Returns the fps of the AFNI file."""
    if not isinstance(self._nib_img, nib.brikhead.AFNIImage):
      raise RuntimeError('This operation requires an AFNI image!')
    t_unit = self._nib_img.header.info['TAXIS_NUMS'][2]
    time_step = self._nib_img.header.info['TAXIS_FLOATS'][1]
    step_to_fps: dict[int, Callable[[float], float]] = {
        _UNIT_SEC: lambda step: float(1 / step),
        _UNIT_MSEC: lambda step: float(1000 / step),
        _UNIT_HZ: float,
    }
    return step_to_fps[t_unit](time_step)

  @functools.cached_property
  def range_min(self) -> float:
    if not isinstance(self._nib_img, nib.brikhead.AFNIImage):
      raise RuntimeError('This operation requires an AFNI image!')
    mins = self._nib_img.header.info['BRICK_STATS'][::2]
    return float(min(mins))

  @functools.cached_property
  def range_max(self) -> float:
    if not isinstance(self._nib_img, nib.brikhead.AFNIImage):
      raise RuntimeError('This operation requires an AFNI image!')
    maxs = self._nib_img.header.info['BRICK_STATS'][1::2]
    return float(max(maxs))
