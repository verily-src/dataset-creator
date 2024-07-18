"""Tests for video_io.py."""

import os
from typing import Optional

from absl.testing import parameterized  # type: ignore[import]
import numpy as np

from dataset_creator.features.video_io import test_utils
from dataset_creator.features.video_io import video_io

_EXPECTED_VIDEO_LENGTH_MILLIS = 212033
_EXPECTED_FRAMES_NUMBER = 6361

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def mock_video_path() -> str:
  return os.path.join(THIS_DIR, 'testdata/test_video.mp4')


def _expected_frame_from_reader(frame_number: int) -> np.ndarray:
  return test_utils.get_frames_array()[frame_number]


class VideoFileIOTest(parameterized.TestCase):

  def test_reader_close(self):
    reader = video_io.VideoFileReader(mock_video_path())
    reader.close()

  def test_raises_error_for_bad_filename(self):
    with self.assertRaises(FileNotFoundError):
      video_io.VideoFileReader('bad_filename')

  def test_raises_error_for_corrupted_video(self):
    # File which is not a video is the same as corrupted video
    with self.assertRaises(IOError):
      video_io.VideoFileReader(test_utils.get_frames_array_path())

  def test_read_local_mp4_file(self):
    with video_io.VideoFileReader(mock_video_path()) as reader:
      frames = list(reader.frames())
      self.assertLen(frames, 30)
      self.assertEqual(frames[0].shape, (400, 600, 3))

      self.assertEqual(reader.frame_count, 30)
      self.assertEqual(reader.fps, 30)
      self.assertEqual(reader.duration_in_millis, 1000)
      self.assertEqual(reader.width, 600)
      self.assertEqual(reader.height, 400)
      self.assertEqual(reader.bitrate, 638)

  def test_read_specific_time(self):
    with video_io.VideoFileReader(mock_video_path()) as reader:
      frame = reader.read_full_resolution_by_timestamp(135)

    test_utils.assert_frames_almost_identical(
        frame, _expected_frame_from_reader(4)
    )

  def test_read_specific_frame(self):
    with video_io.VideoFileReader(mock_video_path()) as reader:
      frame = reader.read_full_resolution_by_frame_number(4)

    test_utils.assert_frames_almost_identical(
        frame, _expected_frame_from_reader(4)
    )

  def test_read_last_frame(self):
    with video_io.VideoFileReader(mock_video_path()) as reader:
      frame_count = reader.frame_count
      frame = reader.read_full_resolution_by_frame_number(frame_count - 1)
      with self.assertRaises(ValueError):
        reader.read_full_resolution_by_frame_number(frame_count)

    test_utils.assert_frames_almost_identical(
        frame, _expected_frame_from_reader(frame_count - 1),
    )

  def test_read_all_frames_after_read_specific_frame(self):
    with video_io.VideoFileReader(mock_video_path()) as reader:
      reader.read_full_resolution_by_frame_number(4)
      frames = list(reader.frames())
      self.assertLen(frames, 30)

  @parameterized.named_parameters(
      ('negative_timestamp', -1, None),
      ('too_big_timestamp', _EXPECTED_VIDEO_LENGTH_MILLIS + 1, None),
      ('negative_frame_number', None, -1),
      ('too_big_frame_number', None, _EXPECTED_FRAMES_NUMBER + 1),
      ('no_parameters', None, None),
  )
  def test_read_at_validation(
      self,
      timestamp_millis: Optional[int],
      frame_number: Optional[int],
  ):
    with video_io.VideoFileReader(mock_video_path()) as reader:
      with self.assertRaises(ValueError):
        reader.read_at(
            timestamp_millis=timestamp_millis, frame_number=frame_number)

  def test_resize(self):
    with video_io.VideoFileReader(mock_video_path()) as reader:
      self.assertEqual(
          reader.read_at(frame_number=0, resize_to=(112, 224)).shape,
          (112, 224, 3))

  def test_read_at_with_both_timestamp_and_frame_number_raises_valueerror(self):
    with video_io.VideoFileReader(mock_video_path()) as reader:
      with self.assertRaisesRegex(ValueError, 'frame_number and timestamp'):
        reader.read_at(timestamp_millis=0, frame_number=0)
