"""A module implementing most of the functionality for an fMRI reader."""

from __future__ import annotations

import abc
import datetime
import functools
import json
from typing import Iterable, Union

import nibabel as nib
import numpy as np

from dataset_creator.features.video_io import base_video_io

_NibImage = Union[nib.Nifti1Image, nib.brikhead.AFNIImage]


def _validate_slice(img: _NibImage, z_slice: slice):
  n_slices = img.header.get_data_shape()[2]
  indices = z_slice.indices(n_slices)
  if indices[0] >= indices[1] or indices[0] < 0:
    raise IndexError('Invalid z_slice.')


def _normalize_images(
    image: np.ndarray, range_min: float, range_max: float
) -> np.ndarray:
  """Normalizes float type into uint16."""
  image_dtype = image.dtype
  if image_dtype in [np.uint8, np.uint16, np.int16]:
    return image
  assert image_dtype in [np.float16, np.float32, np.float64]
  scaled_img = (image - range_min) / (range_max - range_min)
  return (65535 * scaled_img).astype(np.uint16)


class NibabelReader(base_video_io.AbstractVideoReader, abc.ABC):
  """A reader for fMRI images supported by nibaberl.

  Attributes:
    video_path: The path of the file this reader reads from.

  Raises:
    FileNotFoundError: If check_path and the file does not exist.
    ImageFileError: If check_path and the file is not a valid imaging file.
    IndexError: If the z_slice is invalid.
  """

  def __init__(
      self, video_path: str, z_slice: slice = slice(None), **kwargs
  ):
    super().__init__(video_path, **kwargs)
    self._z_slice = z_slice
    self._current_frame_number = 0
    if self._check_path:
      _ = self._nib_img

  @functools.cached_property
  def _nib_img(self) -> _NibImage:
    # Validate the slice before loading the image. This allows us to only
    # validate the slice once, as this is a cached property. This is the right
    # place to validate the slice, as we will read a slice in most cases after
    # this.
    img: _NibImage = nib.load(self._video_path)  # type: ignore[assignment]
    _validate_slice(img, self._z_slice)
    return img

  def read(self) -> base_video_io.Frame:
    """Reads a frame from the reader.

    Returns:
      A frame, which is next one to seek from the reader.

    Raises:
      FileNotFoundError: If check_path is False and the file does not exist.
      ImageFileError: If check_path is False and the file is not a valid imaging
        file.
      IndexError: If check_path is False and the z_slice is invalid.
    """
    img = self.read_at(frame_number=self._current_frame_number)
    self._current_frame_number += 1
    return img

  def _frames_full_resolution(
      self, start_frame_number: base_video_io.FrameNumber = 0
  ) -> Iterable[base_video_io.Frame]:
    """Returns an iterable of all remaining frames in video.

    Args:
      start_frame_number: The frame number in video to start reading from.

    Raises:
      IOError if the reader was never opened or a frame could not be read.
    """
    images = self._nib_img.dataobj[..., self._z_slice, start_frame_number:]
    return _normalize_images(
        images.transpose(3, 0, 1, 2), self.range_min, self.range_max
    )

  def read_full_resolution_by_frame_number(
      self, frame_number: base_video_io.FrameNumber
  ) -> base_video_io.Frame:
    """Returns the 'frame_number' frame in video.

    Args:
      frame_number: The frame number in video to read.

    Raises:
      FileNotFoundError: If check_path is False and the file does not exist.
      ImageFileError: If check_path is False and the file is not a valid imaging
        file.
      IndexError: If check_path is False and the z_slice is invalid.
    """
    image = self._nib_img.dataobj[..., self._z_slice, frame_number]
    return _normalize_images(image, self.range_min, self.range_max)

  def read_full_resolution_by_timestamp(
      self, timestamp_millis: base_video_io.TimestampMilliseconds
  ) -> base_video_io.Frame:
    """Returns the frame from video in time 'timestamp_millis'.

    Args:
      timestamp_millis: The time in millisecond of the frame in video to read.

    Raises:
      FileNotFoundError: If check_path is False and the file does not exist.
      ImageFileError: If check_path is False and the file is not a valid imaging
        file.
      IndexError: If check_path is False and the z_slice is invalid.
    """
    return self.read_at(frame_number=int(self.fps * timestamp_millis / 1000))

  @functools.cached_property
  def n_slices(self):
    """Returns the total depth of the video."""
    return self._nib_img.header.get_data_shape()[2]

  def close(self):
    """Closes the reader."""

  @functools.cached_property
  @abc.abstractmethod
  def fps(self) -> float:
    """Returns the fps of the imaging file."""

  @functools.cached_property
  def range_min(self) -> float:
    """Returns the maximum value of the imaging file.

    This is only used when the dtype of the image is float.
    """
    return self._nib_img.get_fdata().min()

  @functools.cached_property
  def range_max(self) -> float:
    """Returns the maximum value of the imaging file.

    This is only used when the dtype of the image is float.
    """
    return self._nib_img.get_fdata().max()

  @functools.cached_property
  def frame_count(self) -> int:
    """Returns the number of frames in the fMRI file."""
    return self._nib_img.header.get_data_shape()[-1]

  @functools.cached_property
  def duration(self) -> datetime.timedelta:
    """Returns the duration of video."""
    return self.frame_count * datetime.timedelta(seconds=1 / self.fps)

  def serialize(self) -> bytes:
    z_is = [self._z_slice.start, self._z_slice.stop, self._z_slice.step]
    return f'{self._video_path}|{self._check_path}|{json.dumps(z_is)}'.encode()

  @classmethod
  def deserialize(cls, serialized: bytes) -> NibabelReader:
    path, check_path, z_slice = serialized.decode().split('|')
    return cls(
        path,
        check_path=check_path == 'True',
        z_slice=slice(*json.loads(z_slice)),
    )
