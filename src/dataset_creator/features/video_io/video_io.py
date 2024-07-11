"""VideoReader implementations."""

from dataset_creator.features.video_io import base_video_io
from dataset_creator.features.video_io import brik_reader
from dataset_creator.features.video_io import image_reader
from dataset_creator.features.video_io import nii_reader
from dataset_creator.features.video_io import opencv_video_reader
from dataset_creator.features.video_io import youtube_reader

# The video readers and writers here should be used across endoscopy to allow
# for easy replacement of the underlying implementation.

VideoFileReader = opencv_video_reader.OpenCVVideoFileReader
assert issubclass(VideoFileReader, base_video_io.AbstractVideoReader)

NiiReader = nii_reader.NiiReader
assert issubclass(NiiReader, base_video_io.AbstractVideoReader)

BrikReader = brik_reader.BrikReader
assert issubclass(BrikReader, base_video_io.AbstractVideoReader)

ImageReader = image_reader.ImageReader
assert issubclass(ImageReader, base_video_io.AbstractVideoReader)

YoutubeReader = youtube_reader.YoutubeReader
assert issubclass(YoutubeReader, base_video_io.AbstractVideoReader)

IMAGE_READER_EXTENSIONS = image_reader.IMAGE_READER_EXTENSIONS
