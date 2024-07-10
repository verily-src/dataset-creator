"""Tests for brik_reader.py."""

import os

from absl.testing import absltest  # type: ignore[import]

from dataset_creator.features.video_io import brik_reader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(THIS_DIR, 'testdata/fmri+orig.HEAD')
TEST_NII_FILE = os.path.join(THIS_DIR, 'testdata/fmri.nii.gz')


class BrikReaderTest(absltest.TestCase):

  def test_fps(self):
    self.assertEqual(brik_reader.BrikReader(TEST_FILE).fps, 0.5)

  def test_range_min(self):
    self.assertEqual(brik_reader.BrikReader(TEST_FILE).range_min, 0.0)

  def test_range_max(self):
    self.assertEqual(brik_reader.BrikReader(TEST_FILE).range_max, 1643.0)

  def test_fps_raises_with_invalid_file(self):
    with self.assertRaisesRegex(RuntimeError, 'requires an AFNI image'):
      _ = brik_reader.BrikReader(TEST_NII_FILE).fps

  def test_range_min_raises_with_invalid_file(self):
    reader = brik_reader.BrikReader(TEST_NII_FILE, check_path=False)
    with self.assertRaisesRegex(RuntimeError, 'requires an AFNI image'):
      _ = reader.range_min

  def test_range_max_raises_with_invalid_file(self):
    reader = brik_reader.BrikReader(TEST_NII_FILE, check_path=False)
    with self.assertRaisesRegex(RuntimeError, 'requires an AFNI image'):
      _ = reader.range_max
