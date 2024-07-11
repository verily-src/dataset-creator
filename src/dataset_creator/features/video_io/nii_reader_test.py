"""Tests for nii_reader.py."""

import os

from absl.testing import absltest  # type: ignore[import]

from dataset_creator.features.video_io import nii_reader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(THIS_DIR, 'testdata/fmri.nii.gz')
TEST_BRIK_FILE = os.path.join(THIS_DIR, 'testdata/fmri+orig.HEAD')


class NiiReaderTest(absltest.TestCase):

  def test_fps(self):
    self.assertEqual(nii_reader.NiiReader(TEST_FILE).fps, 0.5)

  def test_fps_raises_with_invalid_file(self):
    with self.assertRaisesRegex(RuntimeError, 'operation requires a Nifti'):
      _ = nii_reader.NiiReader(TEST_BRIK_FILE).fps
