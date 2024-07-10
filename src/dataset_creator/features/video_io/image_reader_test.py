"""Tests for image_reader.py."""

import os
import random
import tempfile

from absl.testing import parameterized  # type: ignore[import]
import cv2

from dataset_creator.features.video_io import image_reader
from dataset_creator.features.video_io import test_utils

_IMAGE_PATH = ''
_FRAME_NUMBER = 5


def setUpModule() -> None:
  global _IMAGE_PATH
  with tempfile.NamedTemporaryFile('wb', suffix='.jpg', delete=False) as fn:
    _IMAGE_PATH = fn.name
    frame = test_utils.get_frames_array()[_FRAME_NUMBER]
    cv2.imwrite(_IMAGE_PATH, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def tearDownModule() -> None:
  os.remove(_IMAGE_PATH)


def _get_image_reader() -> image_reader.ImageReader:
  return image_reader.ImageReader(_IMAGE_PATH)


class ImageReaderTest(parameterized.TestCase):

  def test_num_frames(self):
    self.assertEqual(_get_image_reader().frame_count, 1)

  def test_read_at(self):
    reader = _get_image_reader()
    expected_frame = test_utils.get_frames_array()[_FRAME_NUMBER]
    # Check that frame timestamp and frame number are ignored.
    frame = reader.read_at(timestamp_millis=random.randint(0, 99999))
    test_utils.assert_frames_almost_identical(frame, expected_frame)
    frame = reader.read_at(frame_number=random.randint(0, 99999))
    test_utils.assert_frames_almost_identical(frame, expected_frame)

  def test_fps(self):
    self.assertEqual(_get_image_reader().fps, 1)

  @parameterized.named_parameters(
      ('valid_start_frame', 0, 1), ('invalid_start_frame', 5, 0)
  )
  def test_frames(self, start_frame_number: int, expected_length: int):
    self.assertLen(
        list(_get_image_reader().frames(start_frame_number)), expected_length
    )
