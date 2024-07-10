"""Implementation of VideoFileReader for OpenCV."""

import datetime
import functools
from typing import Any, Iterable

import cv2
import numpy as np

from dataset_creator.features.video_io import base_video_io

TimestampMilliseconds = base_video_io.TimestampMilliseconds
FrameNumber = base_video_io.FrameNumber
Frame = base_video_io.Frame


class OpenCVVideoFileReader(base_video_io.AbstractVideoReader):
  """OpenCV reader that uses FFMPEG as a backend."""

  def __init__(self, video_path: str, **kwargs):
    """Instantiates a new OpenCV video reader.

    Args:
      video_path: The path to the video.
      **kwargs: Any additional kwargs to be passed to the AbstractVideoReader.

    Raises:
      IOError: if check_path and a reader for video_path cannot be opened.
    """
    self._cap_initialized = False
    super().__init__(video_path, **kwargs)
    if self._check_path:
      _ = self._video_capture  # Try to open the video and raise upon failure.

  @functools.cached_property
  def _video_capture(self) -> cv2.VideoCapture:  # pylint: disable=method-hidden
    cap = cv2.VideoCapture(self._video_path, apiPreference=cv2.CAP_FFMPEG)
    if not cap.isOpened():
      raise IOError(f'Unable to open video at: {self._video_path}')
    self._cap_initialized = True
    return cap

  def _raise_if_closed(self):
    # We're about to perform I/O so initialize video_capture if not initialized.
    _ = self._video_capture
    if not self._cap_initialized or self._video_capture is None:
      raise IOError('I/O operation on closed file.')

  def _set_param(self, cv2_param: Any, value: Any):
    assert self._video_capture.set(cv2_param, value)

  def read(self) -> np.ndarray:
    """Reads an RGB frame from the reader."""
    self._raise_if_closed()
    status, frame = self._video_capture.read()
    assert status
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  def _frames_full_resolution(
      self, start_frame_number: FrameNumber = 0
  ) -> Iterable[Frame]:
    """Returns an iterable of all remaining frames in video."""
    self._raise_if_closed()
    self._validate_in_range_inclusive(
        start_frame_number, start=0, end=self.frame_count - 1)
    self._set_param(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    for _ in range(self.frame_count - start_frame_number - 1):
      yield self.read()
    yield self.read_full_resolution_by_frame_number(self.frame_count - 1)

  def read_full_resolution_by_frame_number(
      self, frame_number: FrameNumber) -> Frame:
    """Returns the 'frame_number' frame in video."""
    self._raise_if_closed()
    self._validate_in_range_inclusive(
        frame_number, start=0, end=self.frame_count - 1)
    self._set_param(cv2.CAP_PROP_POS_FRAMES, frame_number)
    return self.read()

  def read_full_resolution_by_timestamp(
      self, timestamp_millis: TimestampMilliseconds) -> Frame:
    """Returns the frame from video in time 'timestamp_millis'."""
    self._raise_if_closed()
    self._validate_in_range_inclusive(timestamp_millis,
                                      start=0, end=self.duration_in_millis)
    self._set_param(cv2.CAP_PROP_POS_MSEC, timestamp_millis)
    return self.read()

  def close(self):
    """Closes reader and frees memory."""
    if self._cap_initialized:
      self._video_capture.release()
      self._video_capture = None
      self._cap_initialized = False

  @functools.cached_property
  def fps(self) -> float:
    """Returns the fps of the video."""
    self._raise_if_closed()
    return self._video_capture.get(cv2.CAP_PROP_FPS)

  @functools.cached_property
  def bitrate(self) -> float:
    """Returns the bitrate of the video in kbits/s."""
    self._raise_if_closed()
    return self._video_capture.get(cv2.CAP_PROP_BITRATE)

  @functools.cached_property
  def width(self) -> int:
    """Returns the width of frames in the video."""
    self._raise_if_closed()
    return int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

  @functools.cached_property
  def height(self) -> int:
    """Returns the height of frames in the video."""
    self._raise_if_closed()
    return int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

  @functools.cached_property
  def frame_count(self) -> int:
    """Returns the number of frames in the video."""
    self._raise_if_closed()
    return int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

  @functools.cached_property
  def duration(self) -> datetime.timedelta:
    """Returns the duration of video."""
    self._raise_if_closed()
    return datetime.timedelta(seconds=self.frame_count / self.fps)
