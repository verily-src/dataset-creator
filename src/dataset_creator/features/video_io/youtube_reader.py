"""A module implementing a reader from YouTube."""

import yt_dlp  # type: ignore[import]

from dataset_creator.features.video_io import opencv_video_reader


class YoutubeReader(opencv_video_reader.OpenCVVideoFileReader):
  def __init__(self, url: str, **kwargs):
    options = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(options) as dl:
      info = dl.extract_info(url, download=False)
    if 'check_path' in kwargs:
      kwargs.pop('check_path')
    super().__init__(info['url'], check_path=False, **kwargs)
