"""Tests for opencv_video_reader.py."""

import os
import tempfile

from absl.testing import absltest  # type: ignore[import]

from dataset_creator.features.video_io import opencv_video_reader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_NAME = 'testdata/test_video.mp4'


def _test_filename() -> str:
  return os.path.join(THIS_DIR, TEST_FILE_NAME)


class OpenCVVideoReaderTest(absltest.TestCase):

  def test_reader_close(self):
    reader = opencv_video_reader.OpenCVVideoFileReader(_test_filename())
    reader.close()
    # pylint: disable-next=protected-access,consider-using-with
    self.assertIsNone(reader._video_capture)

  def test_raises_error_for_bad_filename(self):
    with self.assertRaises(FileNotFoundError):
      opencv_video_reader.OpenCVVideoFileReader('bad_filename')

  def test_frames_with_resizing(self):
    reader = opencv_video_reader.OpenCVVideoFileReader(_test_filename())
    frames = list(
        reader.frames(
            start_frame_number=reader.frame_count - 5,
            resize_to=(10, 10)
        )
    )
    self.assertEqual(len(frames), 5)
    self.assertEqual(frames[0].shape, (10, 10, 3))

  def test_raises_ioerror_on_invalid_file(self):
    with tempfile.NamedTemporaryFile() as tmpfile:
      tmpfile.write(b'Invalid video format')
      with self.assertRaisesRegex(IOError, 'Unable to open video at'):
        opencv_video_reader.OpenCVVideoFileReader(tmpfile.name, check_path=True)

  def test_read_after_close_raises_ioerror(self):
    reader = opencv_video_reader.OpenCVVideoFileReader(_test_filename())
    reader.close()
    with self.assertRaisesRegex(IOError, 'I/O operation on closed file'):
      reader.read_at(frame_number=0)

  def test_read_at_invalid_frame_number_raises_valueerror(self):
    reader = opencv_video_reader.OpenCVVideoFileReader(_test_filename())
    with self.assertRaisesRegex(ValueError, '-1 is invalid.'):
      reader.read_full_resolution_by_frame_number(frame_number=-1)
