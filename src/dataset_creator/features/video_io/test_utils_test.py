"""Tests for nii_reader.py."""

from absl.testing import absltest  # type: ignore[import]
import numpy as np

from dataset_creator.features.video_io import test_utils


class YoutubeReaderTest(absltest.TestCase):

  def test_assert_frames_almost_identical_raise_with_distinct_frames(self):
    with self.assertRaisesRegex(AssertionError, 'Frame is not equal'):
      test_utils.assert_frames_almost_identical(
          np.zeros(5, dtype=np.uint8),
          100 * np.ones(5, dtype=np.uint8)
      )
