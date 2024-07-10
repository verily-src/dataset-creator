"""Tests for nii_reader.py."""

import os

from absl.testing import parameterized  # type: ignore[import]
import nibabel as nib
import numpy as np
import tensorflow as tf

from dataset_creator.features.video_io import brik_reader
from dataset_creator.features.video_io import nii_reader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(THIS_DIR, 'testdata/fmri.nii.gz')
TEST_BRIK_FILE = os.path.join(THIS_DIR, 'testdata/fmri+orig.HEAD')


class NiiReaderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = nii_reader.NiiReader(TEST_FILE)

  @parameterized.named_parameters(
      ('non_existent_file', 'non_existent_file', FileNotFoundError),
      (
          'non_fmri_file',
          os.path.join(THIS_DIR, 'testdata/test_video.mp4'),
          nib.filebasedimages.ImageFileError
      ),
  )
  def test_instantiation_raises_on_invalid_file_with_check_path(
      self, path: str, exception_type: type
  ):
    with self.assertRaises(exception_type):
      nii_reader.NiiReader(path, check_path=True)

  @parameterized.named_parameters(
      ('non_existent_file', 'non_existent_file'),
      ('non_fmri_file', os.path.join(THIS_DIR, 'testdata/test_video.mp4')),
  )
  def test_instantiation_doesnt_raise_on_invalid_file_without_check_path(
      self, path: str
  ):
    nii_reader.NiiReader(path, check_path=False)

  def test_instantiation_raises_on_invalid_z_slice(self):
    z_slice = slice(-1, 1)
    with self.assertRaisesRegex(IndexError, 'Invalid z_slice.'):
      nii_reader.NiiReader(TEST_FILE, z_slice=z_slice, check_path=True)

  def test_read(self):
    self.assertEqual(self.reader.read().shape, (64, 64, 37))

  def test_frames(self):
    frames = list(self.reader.frames(start_frame_number=5))
    self.assertLen(frames, 36 - 5)

  def test_n_slices(self):
    self.assertEqual(self.reader.n_slices, 37)

  def test_duration(self):
    self.assertEqual(self.reader.duration.total_seconds(), 72)

  def test_close_not_raises(self):
    self.reader.close()

  def test_read_by_timestamp_reads_the_correct_frame(self):
    frame = self.reader.read_at(timestamp_millis=3500)
    expected_frame = self.reader.read_at(frame_number=1)
    self.assertAllEqual(frame, expected_frame)

  def test_deserialize_returns_a_reader_with_the_same_z_slice(self):
    # pylint: disable=protected-access
    z_slice = slice(0, 10)
    reader = nii_reader.NiiReader(TEST_FILE, z_slice=z_slice)
    reader_range = reader._z_slice.indices(reader.n_slices)
    deserialized_reader = nii_reader.NiiReader.deserialize(reader.serialize())
    deserialized_reader_range = reader._z_slice.indices(
        deserialized_reader.n_slices
    )
    self.assertEqual(reader_range, deserialized_reader_range)
    # pylint: enable=protected-access

  def test_serialize_doesnt_perform_io(self):
    reader = nii_reader.NiiReader(
        'path_doesnt_exist.nii.gz', check_path=False
    )
    self.assertIsInstance(reader.serialize(), bytes)

  def test_reader_always_returns_uint_dtype(self):
    reader = brik_reader.BrikReader(TEST_BRIK_FILE)
    self.assertEqual(reader.read_at(frame_number=0).dtype, np.uint16)
