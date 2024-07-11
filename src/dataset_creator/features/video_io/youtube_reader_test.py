"""Tests for nii_reader.py."""

from absl.testing import absltest  # type: ignore[import]

from dataset_creator.features.video_io import youtube_reader


class YoutubeReaderTest(absltest.TestCase):

  def test_check_path_is_overriden(self):
    youtube_reader.YoutubeReader(
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ', check_path=True,
    )
