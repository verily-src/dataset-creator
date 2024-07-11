"""Abstract classes for VideoReader."""

from __future__ import annotations

import abc
from collections.abc import Iterable
from collections.abc import Sequence
import datetime
import functools
import os

import cv2
import more_itertools
import numpy as np

from dataset_creator.features import serializable

TimestampMilliseconds = int
TimestampsSequence = Sequence[TimestampMilliseconds]
FrameNumber = int
FrameNumberSequence = Sequence[int]
Frame = np.ndarray
FramesSequence = Sequence[np.ndarray]
FrameSize = tuple[int, int]


def _resize_frames_if_needed(
    frames: Iterable[Frame], resize_to: FrameSize | None) -> Iterable[Frame]:
  if resize_to is not None:
    # OpenCV accepts (width, height), so flip the order
    return (cv2.resize(frame, dsize=resize_to[::-1]) for frame in frames)
  return frames


class AbstractVideoReader(serializable.Serializable, abc.ABC):
  """Abstract base class for video file readers.

  Attributes:
    video_path: The path of the file this reader reads from.
  """

  def __init__(self, video_path: str, check_path: bool = True):
    """Initiates a VideoReader.

    Args:
      video_path: Path to a video.
      check_path: Check whether video_path exists and contains a valid file.

    Raises:
      FileNotFoundError: if check_path and the file does not exist.
    """
    if check_path and not os.path.exists(video_path):
      raise FileNotFoundError(f'{video_path} does not exist')

    self._video_path = video_path
    self._check_path = check_path

  @abc.abstractmethod
  def read(self) -> Frame:
    """Reads a frame from the reader.

    Returns:
      A frame, which is next one to seek from the reader.

    Raises:
      IOError if the reader was never opened or a frame could not be read.
    """

  def frames(
    self,
    start_frame_number: FrameNumber = 0,
    *,
    resize_to: FrameSize | None = None
  ) -> Iterable[Frame]:
    """Returns an iterable of all remaining frames in video, possibly resized.

    Args:
      start_frame_number: The frame number in video to start reading from.
      resize_to: Resize to this size. Input is (H, W). Default is don't resize.

    Raises:
      IOError if the reader was never opened or a frame could not be read.
    """
    return _resize_frames_if_needed(
        self._frames_full_resolution(start_frame_number), resize_to
    )

  @abc.abstractmethod
  def _frames_full_resolution(
      self, start_frame_number: FrameNumber = 0
  ) -> Iterable[Frame]:
    """Returns an iterable of all remaining frames in video.

    Args:
      start_frame_number: The frame number in video to start reading from.

    Raises:
      IOError if the reader was never opened or a frame could not be read.
    """

  def read_at(
      self,
      *,
      timestamp_millis: TimestampMilliseconds | None = None,
      frame_number: FrameNumber | None = None,
      resize_to: FrameSize | None = None
  ) -> Frame:
    """Returns an RGB Frame from timestamp in milliseconds or the frame number.

    Args:
      timestamp_millis: A timestamp to read from.
      frame_number: A frame number to read.
      resize_to: Resize to this size. Input is (H, W). Default is don't resize.

    Raises:
      ValueError:  In case bot timestamp and frame_number are provided or no
      timestamp nor frame_number are provided.
    """
    if timestamp_millis is not None and frame_number is not None:
      raise ValueError('Can\'t provide both frame_number and timestamp_millis')
    if timestamp_millis is None and frame_number is None:
      raise ValueError(
        'Frame number must be >= 0 when timestamp_millis is None')
    if timestamp_millis is not None:
      self._validate_in_range_inclusive(timestamp_millis, 0,
                                        self.duration_in_millis)
      frame = self.read_full_resolution_by_timestamp(timestamp_millis)
    else:
      assert frame_number is not None
      # Sample according to the frame number
      self._validate_in_range_inclusive(frame_number, 0, self.frame_count)
      frame = self.read_full_resolution_by_frame_number(frame_number)
    return more_itertools.first(_resize_frames_if_needed([frame], resize_to))

  @abc.abstractmethod
  def read_full_resolution_by_frame_number(
      self, frame_number: FrameNumber) -> Frame:
    """Returns the 'frame_number' frame in video.

    Args:
      frame_number: The frame number in video to read.

    Raises:
      IOError if the reader was never opened or a frame could not be read.
    """

  @abc.abstractmethod
  def read_full_resolution_by_timestamp(
      self, timestamp_millis: TimestampMilliseconds) -> Frame:
    """Returns the frame from video in time 'timestamp_millis'.

    Args:
      timestamp_millis: The time in millisecond of the frame in video to read.

    Raises:
      IOError if the reader was never opened or a frame could not be read.
    """

  def _validate_in_range_inclusive(self, value, start, end):
    if value < start or value > end:
      raise ValueError(
        f'{value} is invalid. {self._video_path} start = {start}, end = {end}'
      )

  @abc.abstractmethod
  def close(self):
    """Closes the reader."""

  @functools.cached_property
  @abc.abstractmethod
  def fps(self) -> float:
    """Returns the fps of the video."""

  @functools.cached_property
  @abc.abstractmethod
  def frame_count(self) -> int:
    """Returns the number of frames in the video."""

  @functools.cached_property
  @abc.abstractmethod
  def duration(self) -> datetime.timedelta:
    """Returns the duration of video."""

  @functools.cached_property
  def duration_in_millis(self) -> float:
    """Returns the duration of video in milliseconds."""
    return self.duration / datetime.timedelta(milliseconds=1)

  @property
  def video_path(self) -> str:
    return self._video_path

  def __enter__(self):
    """Allows the reader to be used in a with-statement context."""
    return self

  def __exit__(self, *args):
    """Allows the reader to be used in a with-statement context."""
    self.close()

  def __del__(self):
    """Closes the reader."""
    self.close()

  def serialize(self) -> bytes:
    return f'{self._video_path},{self._check_path}'.encode()

  @classmethod
  def deserialize(cls, serialized: bytes) -> AbstractVideoReader:
    path, check_path = serialized.decode().split(',')
    return cls(path, check_path=check_path == 'True')
