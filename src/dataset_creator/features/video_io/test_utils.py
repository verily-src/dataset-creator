"""Utils for tests related to video_io."""

import functools
import os

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPECTED_FRAMES_FROM_READER_PATH = 'testdata/expected_frames_from_reader.npy'


def are_images_almost_identical(first: np.ndarray, second: np.ndarray) -> bool:
  """Checks if two images are identical and ignores un-visible differences.

  This method can be used to check equality of videos produced by a video writer
  as usually the produced videos are visibly the same but the numeric value of
  each pixel might vary.

  Args:
    first: An image tensor.
    second: An image tensor.

  Returns:
    Whether the two images are almost identical.
  """
  # Convert to int instead of uint8 to avoid underflows.
  first = first.astype(int)
  second = second.astype(int)
  diff = np.abs(first - second)
  return diff.max() < 45 and diff.mean() < 3


def assert_frames_almost_identical(
    first: np.ndarray,
    second: np.ndarray,
):
  """Raises AssertionError on visibly unidentical frames."""
  if are_images_almost_identical(first, second):
    return
  raise AssertionError('Frame is not equal to the expected frame')


def get_frames_array_path() -> str:
  return os.path.join(THIS_DIR, _EXPECTED_FRAMES_FROM_READER_PATH)


@functools.lru_cache()
def get_frames_array() -> np.ndarray:
  return np.load(get_frames_array_path())
