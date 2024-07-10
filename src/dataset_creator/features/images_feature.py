"""A module for reading images / frames from videos."""

import functools
import os
from typing import Any, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import numpy.typing  # pylint: disable=unused-import
import tensorflow as tf
from typing_extensions import TypeAlias

from dataset_creator.features import base_feature
from dataset_creator.features import fields
from dataset_creator.features.video_io import base_video_io
from dataset_creator.features.video_io import video_io

_TimestampMillis: TypeAlias = base_video_io.TimestampMilliseconds
_FrameNumber: TypeAlias = base_video_io.FrameNumber
_Frame: TypeAlias = base_video_io.Frame
_FrameOrFrames: TypeAlias = Union[_Frame, Sequence[_Frame]]

READ_BY_FRAME_NUMBER = 'frame_number'
READ_BY_TIMESTAMP_MILLIS = 'timestamp_millis'
READ_ALL_FRAMES = 'all_frame_numbers'


def get_default_reader(
    path: str, **kwargs
) -> base_video_io.AbstractVideoReader:
  """Returns a reader instance to read frames from video_path."""
  _, ext = os.path.splitext(path)
  if ext in video_io.IMAGE_READER_EXTENSIONS:
    return video_io.ImageReader(path, **kwargs)
  if ext == '.nii' or path.endswith('.nii.gz'):
    return video_io.NiiReader(path, **kwargs)
  if ext.upper() in ['.HEAD', '.BRIK']:
    return video_io.BrikReader(path, **kwargs)
  return video_io.VideoFileReader(path, **kwargs)


class ImagesFeature(base_feature.CustomFeature):
  """A feature that converts a video reader and timestamps to frames."""

  _serializable_classes = [
      video_io.BrikReader,
      video_io.ImageReader,
      video_io.NiiReader,
      video_io.VideoFileReader,
      video_io.YoutubeReader,
  ]

  def __init__(
      self,
      *,
      reader: base_video_io.AbstractVideoReader,
      read_by: str,
      read_at: Sequence[int],
      image_size: Optional[base_video_io.FrameSize] = None,
      num_dims: int = 2,
      **kwargs,
  ):
    """Instantiate an ImagesFeature.

    Args:
      reader: The AbstractVideoReader to use for reading in this feature.
      read_by: Can only take three values - READ_BY_FRAME, READ_ALL_FRAMES or
        READ_BY_TIMESTAMP_MILLIS. If READ_BY_FRAME, frames are read by frame
        number from reat_at. If READ_BY_TIMESTAMP, frames are read by
        timestamp_millis from read_at. Please note that when using
        READ_ALL_FRAMES, the job is not split, so take note of using a reader
        that has an optimized .frames method.
      read_at: The timestamps or frame_numbers to read from the reader. When
        read_by is READ_ALL_FRAMES, read_at should have only one element, which
        is the start frame number to read from.
      image_size: Resize read images to this size.
      num_dims: The number of dimension for each read image. Possible values are
        2 or 3. Default is 2. At the moment, 2-D images are assumed to always be
        RGB, while 3-D images are assumed to always be grayscale.
      **kwargs: Additional keyword arguments to be passed to CustomFeature.
    """

    if not read_at:
      raise ValueError('At least one timestamp or frame number is required.')
    if read_by == READ_ALL_FRAMES and len(read_at) != 1:
      raise ValueError(
          'When reading all frames, read_at should only include the start frame'
          ' to read from.'
      )
    if read_by not in [
        READ_BY_FRAME_NUMBER, READ_BY_TIMESTAMP_MILLIS, READ_ALL_FRAMES
    ]:
      raise ValueError('Only frame_number and timestamp_millis are allowed.')
    if num_dims not in [2, 3]:
      raise ValueError('num_dims must be either 2 or 3.')

    super().__init__(
        reader=reader,
        read_by=read_by,
        read_at=read_at,
        image_size=image_size,
        num_dims=num_dims,
        **kwargs,
    )

    self._read_by = read_by
    self._read_at = list(read_at)
    self._reader = reader
    self._image_size = image_size
    self._num_dims = num_dims

  def split(self) -> list[int]:
    """See base class."""
    return self._read_at

  def process(
      self, metadata_value: Union[_TimestampMillis, _FrameNumber], _: Any,
  ) -> Optional[_FrameOrFrames]:
    """Reads the timestamp using self.reader.

    Args:
      metadata_value: The timestamp or frame number to read.

    Returns:
      A frame corresponding to the given timestamp or frame number and
      self.reader, or None in case the read operation failed.
    """
    kwargs = {self._read_by: metadata_value}
    try:
      if self._read_by == READ_ALL_FRAMES:
        return list(
            self._reader.frames(
                start_frame_number=metadata_value, resize_to=self._image_size
            )
        )
      return self._reader.read_at(**kwargs, resize_to=self._image_size)
    except (ValueError, IOError):
      return None

  def merge(
      self, values: Sequence[base_video_io.Frame]
  ) -> dict[str, base_feature.ValueFeature]:
    """Merges the read images and encodes them.

    Args:
      values: The sequence of read images.

    Returns:
      A dictionary containing the following keys:
        fields.IMAGES: The read images. Shape is (N, H, W, Depth / Channels).
        fields.IMAGES_READ_BY: Whether images were read by timestamp or by frame
          number.
        fields.IMAGES_READ_AT: The identifiers (timestamps or frame numbers) to
          read the video.
        fields.IMAGES_ENCODED: The encoded images. Shape is (N, [H])
    """
    if not values:
      return {}
    if self._read_by == READ_ALL_FRAMES:
      assert len(values) == 1
      values = values[0]  # type: ignore[assignment]
    encoded_images: list[Union[bytes, list[bytes]]] = []
    # TODO(itayr): Add support for the cases of grayscale 2D, and RGB 3D images.
    for img in values:
      img = img if img.dtype.type is np.uint8 else img.astype(np.uint16)
      if self._num_dims == 2:
        encoded_images.append(
            cv2.imencode('.png', img[:, :, ::-1])[1].tobytes()
        )
      else:
        assert self._num_dims == 3
        encoded_images.append(
            [
                cv2.imencode('.png', img[i])[1].tobytes()
                for i in range(img.shape[0])
            ]
        )
    return {
        fields.IMAGES: np.stack(values),
        fields.IMAGES_READ_BY: self._read_by,
        fields.IMAGES_READ_AT: self._read_at,
        fields.IMAGES_ENCODED: np.array(encoded_images),
        fields.IMAGES_PATH: self._reader.video_path,
        fields.IMAGES_FPS: self._reader.fps,
        fields.IMAGES_NUM_BITS: values[0].dtype.itemsize * 8,
    }

  @property
  def output_signature(self) -> Mapping[str, Any]:
    return {
        fields.IMAGES_READ_BY: str,
        fields.IMAGES_READ_AT: list[int],
        fields.IMAGES_ENCODED: np.typing.NDArray[np.string_],
        fields.IMAGES_PATH: str,
        fields.IMAGES_FPS: float,
        fields.IMAGES_NUM_BITS: int,
    }


def torch_decode_images(
    element: dict[str, Any]
) -> dict[str, Any]:
  output_element = {}
  for key, value in element.items():
    if not key.endswith(f'/{fields.IMAGES_ENCODED}'):
      output_element[key] = value
      continue
    original_shape = value.shape
    flattened_encodings = value.ravel()

    key_prefix = key[:-len(fields.IMAGES_ENCODED)]
    num_bits_key = key_prefix + fields.IMAGES_NUM_BITS
    dtype = tf.uint16 if element[num_bits_key] == 16 else tf.uint8
    flattened_decoded_images = np.asarray(
        [
            # We cannot use torchvision.io.decode_image as it doesn't support
            # 16-bit images :(
            tf.image.decode_png(encoded_image, dtype=dtype).numpy()
            for encoded_image in flattened_encodings
        ]
    )
    single_image_shape = flattened_decoded_images.shape[1:]

    # Stack all images, but squeeze for the case of only one image:
    new_key = key_prefix + fields.IMAGES
    output_element[new_key] = flattened_decoded_images.reshape(
        original_shape + single_image_shape
    ).squeeze()
  return output_element


@tf.function
def decode_image(
    encoded_image: tf.Tensor, num_bits: tf.Tensor,
) -> tf.Tensor:
  pred_fn_pairs = [
      (
          tf.equal(num_bits, 16),
          lambda: tf.io.decode_png(encoded_image, dtype=tf.uint16)
      )
  ]
  return tf.case(
      pred_fn_pairs,
      default=lambda: tf.cast(tf.image.decode_png(encoded_image), tf.uint16)
  )


@tf.function
def tf_decode_images(element: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
  """Decodes all encoded images in the given element.

  Args:
    element: A mapping possibly containing images to be decoded.

  Returns:
    The same mapping with all images decoded. The decoded images sub-keys are
    fields.IMAGES, while the encoded images are removed from the mapping.
  """
  output_element = {}
  for key, value in element.items():
    if not key.endswith(f'/{fields.IMAGES_ENCODED}'):
      output_element[key] = value
      continue
    original_shape = tf.shape(value)
    flattened_encodings = tf.reshape(value, (-1,))

    key_prefix = key[:-len(fields.IMAGES_ENCODED)]
    num_bits_key = key_prefix + fields.IMAGES_NUM_BITS
    flattened_decoded_images = tf.vectorized_map(
        # Use partial instead of a lambda to avoid a cell-var-from-loop error.
        functools.partial(decode_image, num_bits=element[num_bits_key]),
        flattened_encodings
    )
    single_image_shape = tf.shape(flattened_decoded_images)[1:]
    final_shape = tf.concat([original_shape, single_image_shape], 0)

    # Stack all images, but squeeze for the case of only one image:
    new_key = key_prefix + fields.IMAGES
    output_element[new_key] = tf.squeeze(
        tf.reshape(flattened_decoded_images, final_shape)
    )
  return output_element
