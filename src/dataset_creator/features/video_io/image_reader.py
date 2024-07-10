"""Allows to read an image with the interface of a video reader."""

from collections.abc import Iterable
import datetime
import functools
from typing import Any

import cv2

from dataset_creator.features.video_io import base_video_io

TimestampMilliseconds = base_video_io.TimestampMilliseconds
FrameNumber = base_video_io.FrameNumber
Frame = base_video_io.Frame


IMAGE_READER_EXTENSIONS = ['.jpg', '.jpeg', '.png']


class ImageReader(base_video_io.AbstractVideoReader):
  """A degenerated VideoReader that reads from an image file, not a video."""

  def read_full_resolution_by_timestamp(
      # pylint:disable-next=unused-argument
      self, timestamp_millis: TimestampMilliseconds
  ) -> Frame:
    return self.read()

  def read_full_resolution_by_frame_number(
      # pylint:disable-next=unused-argument
      self, frame_number: FrameNumber
  ) -> Frame:
    return self.read()

  # pylint:disable-next=unused-argument
  def _validate_in_range_inclusive(self, value: Any, start: Any, end: Any):
    # All values are valid, since we don't need to seek to a frame number or
    # timestamp when we only have a single image in hand.
    return

  @functools.cached_property
  def fps(self) -> float:
    return 1.

  @functools.cached_property
  def frame_count(self) -> int:
    return 1

  @functools.cached_property
  def duration(self) -> datetime.timedelta:
    return datetime.timedelta(milliseconds=1)

  def read(self) -> Frame:
    bgr_frame = cv2.imread(self._video_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

  def _frames_full_resolution(
      self, start_frame_number: FrameNumber = 0
  ) -> Iterable[Frame]:
    if start_frame_number in [0, -1]:
      return [self.read()]
    return []

  def close(self):
    pass
