"""Tests for images_feature.py."""

from typing import Sequence

from absl.testing import parameterized  # type: ignore[import]
import cv2
import more_itertools
import numpy as np
import tensorflow as tf
from typing_extensions import TypeAlias

from dataset_creator import test_utils
from dataset_creator.features import fields
from dataset_creator.features import images_feature
from dataset_creator.features.video_io import base_video_io
from dataset_creator.features.video_io import video_io

_TimestampMillis: TypeAlias = base_video_io.TimestampMilliseconds
_FrameNumber: TypeAlias = base_video_io.FrameNumber


def new_feature(
    shape: tuple[int, int],
    timestamps: Sequence[_TimestampMillis] = (),
    frame_numbers: Sequence[_FrameNumber] = (),
) -> images_feature.ImagesFeature:
  if timestamps:
    read_by = images_feature.READ_BY_TIMESTAMP_MILLIS
  else:
    read_by = images_feature.READ_BY_FRAME_NUMBER
  return images_feature.ImagesFeature(
      reader=video_io.VideoFileReader(test_utils.mock_video_path()),
      read_by=read_by,
      read_at=timestamps or frame_numbers,
      image_size=shape,
  )


class ImagesFeatureTest(parameterized.TestCase, tf.test.TestCase):

  def test_raises_with_empty_timestamps(self):
    with self.assertRaises(ValueError):
      new_feature((1, 1), timestamps=(), frame_numbers=())

  @parameterized.named_parameters(
      ('timestamps', [0, 1, 2], []),
      ('frame_numbers', [], [0, 1, 2])
  )
  def test_split(
      self,
      timestamps: Sequence[_TimestampMillis],
      frame_numbers: Sequence[_FrameNumber]
  ):
    feature = new_feature((1, 1), timestamps, frame_numbers)
    self.assertEqual(feature.split(), [0, 1, 2])

  @parameterized.named_parameters(
      ('timestamps', [0, 1, 2], []),
      ('frame_numbers', [], [0, 1, 2])
  )
  def test_process(
      self, timestamps: Sequence[_TimestampMillis],
      frame_numbers: Sequence[_FrameNumber]
  ):
    feature = new_feature((10, 20), timestamps, frame_numbers)
    processed = feature.process(0, None)
    self.assertIsNotNone(processed)
    self.assertEqual(processed.shape, (10, 20, 3))  # type: ignore

  @parameterized.named_parameters(
      ('timestamps', [0, 1, 2], []),
      ('frame_numbers', [], [0, 1, 2])
  )
  def test_process_returns_none_with_invalid_parameter(
      self,
      timestamps: Sequence[_TimestampMillis],
      frame_numbers: Sequence[_FrameNumber]
  ):
    feature = new_feature((1, 1), timestamps, frame_numbers)
    self.assertIsNone(feature.process(-1, None))

  def test_merge(self):
    feature = new_feature((10, 20), [0, 1, 2])
    frames = [np.zeros((10, 20, 3)) for _ in [0, 1, 2]]
    output = feature.merge(frames)
    self.assertEqual(output[fields.IMAGES_READ_AT], [0, 1, 2])
    self.assertEqual(output[fields.IMAGES].shape, (3, 10, 20, 3))
    self.assertNotEmpty(more_itertools.first(output[fields.IMAGES_ENCODED]))

  def test_merge_with_no_frames(self):
    feature = new_feature((1, 1), [0, 1, 2])
    self.assertEmpty(feature.merge([]))

  def test_output_signature_keys_are_a_subset_of_merge_output_keys(self):
    feature = new_feature((10, 20), [0, 1, 2])
    frames = [np.zeros((10, 20, 3)) for _ in [0, 1, 2]]
    merged_keys = feature.merge(frames).keys()
    self.assertContainsSubset(feature.output_signature.keys(), merged_keys)

  @parameterized.named_parameters(('eager', True), ('graph_mode', False))
  def test_tf_decode_images_decodes_as_expected(self, run_eagerly: bool):
    tf.config.run_functions_eagerly(run_eagerly)
    image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    encoded = tf.io.encode_png(tf.convert_to_tensor(image))
    encoded_collection = {
        f'test/{fields.IMAGES_ENCODED}': tf.constant(encoded, shape=(5,)),
        f'test/{fields.IMAGES_NUM_BITS}': 8,
        'label': 1,
    }
    decoded = images_feature.tf_decode_images(encoded_collection)
    self.assertSameElements(
        [f'test/{fields.IMAGES}', f'test/{fields.IMAGES_NUM_BITS}', 'label'],
        decoded
    )
    self.assertAllEqual(decoded[f'test/{fields.IMAGES}'][0], image)

  def test_torch_decode_images_decodes_as_expected(self):
    image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    encoded = tf.io.encode_png(tf.convert_to_tensor(image)).numpy()
    encoded_collection = {
        f'test/{fields.IMAGES_ENCODED}': np.array([encoded for _ in range(5)]),
        f'test/{fields.IMAGES_NUM_BITS}': 8,
        'label': 1,
    }
    decoded = images_feature.torch_decode_images(encoded_collection)
    self.assertSameElements(
        [f'test/{fields.IMAGES}', f'test/{fields.IMAGES_NUM_BITS}', 'label'],
        decoded
    )
    self.assertAllEqual(decoded[f'test/{fields.IMAGES}'][0], image)

  @parameterized.named_parameters(
      ('video', test_utils.mock_video_path(), video_io.VideoFileReader),
      ('image', test_utils.mock_image_path(), video_io.ImageReader),
      ('compressed_fmri', 'test_fmri.nii.gz', video_io.NiiReader),
      ('decompressed_fmri', 'test_fmri.nii', video_io.NiiReader),
      ('head_fmri', 'test_fmri.HEAD', video_io.BrikReader),
      ('brik_fmri', 'test_fmri.BRIK', video_io.BrikReader),
  )
  def test_get_default_reader(
      self,
      path: str,
      expected_cls: type[base_video_io.AbstractVideoReader]
  ):
    reader = images_feature.get_default_reader(path, check_path=False)
    self.assertIsInstance(reader, expected_cls)

  def test_images_feature_raises_on_invalid_read_at(self):
    reader = images_feature.get_default_reader(test_utils.mock_video_path())
    with self.assertRaisesRegex(
        ValueError, 'Only frame_number and timestamp_millis are allowed.'
    ):
      images_feature.ImagesFeature(
          reader=reader, read_by='timestamp_seconds', read_at=[0],
      )

  def test_images_feature_raises_on_invalid_num_dims(self):
    reader = images_feature.get_default_reader(test_utils.mock_video_path())
    with self.assertRaisesRegex(ValueError, '2 or 3'):
      images_feature.ImagesFeature(
          reader=reader,
          read_by=images_feature.READ_BY_FRAME_NUMBER,
          read_at=[0],
          num_dims=1,
      )

  def test_merge_with_3d_images_returns_the_desired_shapes(self):
    reader = images_feature.get_default_reader(test_utils.mock_video_path())
    feature = images_feature.ImagesFeature(
        reader=reader,
        read_by=images_feature.READ_BY_FRAME_NUMBER,
        read_at=range(10),
        num_dims=3,
    )
    merged = feature.merge(
        [np.zeros((100, 200, 30), dtype=np.uint8) for _ in range(10)]
    )
    self.assertEqual(merged[fields.IMAGES].shape, (10, 100, 200, 30))
    self.assertEqual(merged[fields.IMAGES_ENCODED].shape, (10, 100))

  @parameterized.named_parameters(('eager', True), ('graph_mode', False))
  def test_decode_image(self, run_eagerly: bool):
    tf.config.run_functions_eagerly(run_eagerly)
    image = np.random.randint(0, 65535, size=(120, 120, 3), dtype=np.uint16)
    encoded = cv2.imencode('.png', image[..., ::-1])[1].tobytes()
    self.assertAllEqual(images_feature.decode_image(encoded, 16), image)

  def test_merge_with_16bit_images(self):
    feature = new_feature((120, 120), [0])
    images = [np.zeros((10, 10, 3), dtype=np.uint16) for _ in range(3)]
    merged = feature.merge(images)
    self.assertEqual(merged[fields.IMAGES_ENCODED].shape, (3,))

  @parameterized.named_parameters(('eager', True), ('graph_mode', False))
  def test_decode_images_with_16bit_images(self, run_eagerly: bool):
    tf.config.run_functions_eagerly(run_eagerly)
    image = np.random.randint(0, 65535, size=(120, 120, 3), dtype=np.uint16)
    tf_encoded_image = tf.io.encode_png(tf.convert_to_tensor(image))
    encoded_collection = {
        f'test/{fields.IMAGES_ENCODED}': tf.expand_dims(tf_encoded_image, 0),
        f'test/{fields.IMAGES_NUM_BITS}': 16,
    }
    decoded = images_feature.tf_decode_images(encoded_collection)
    self.assertEqual(decoded[f'test/{fields.IMAGES}'].dtype, tf.uint16)
    self.assertAllClose(decoded[f'test/{fields.IMAGES}'], image)

  def test_full_flow_with_read_all_frames(self):
    reader = images_feature.get_default_reader(test_utils.mock_video_path())
    start_from = reader.frame_count - 10
    feature = images_feature.ImagesFeature(
        reader=reader,
        read_by=images_feature.READ_ALL_FRAMES,
        read_at=[start_from],
    )
    processed = [feature.process(n, None) for n in feature.split()]
    merged = feature.merge(processed)
    self.assertLen(merged[fields.IMAGES], 10)
    self.assertEqual(
        merged[fields.IMAGES_READ_BY], images_feature.READ_ALL_FRAMES
    )

  def test_images_feature_raises_on_invalid_read_at_with_read_all_frames(self):
    reader = images_feature.get_default_reader(test_utils.mock_video_path())
    with self.assertRaisesRegex(
        ValueError, 'read_at should only include the start frame'
    ):
      images_feature.ImagesFeature(
          reader=reader, read_by=images_feature.READ_ALL_FRAMES, read_at=[0, 1],
      )

  def test_images_feature_equal(self):
    feature1 = images_feature.ImagesFeature(
        reader=images_feature.get_default_reader(test_utils.mock_video_path()),
        read_by=images_feature.READ_BY_FRAME_NUMBER,
        read_at=[0],
    )
    feature2 = images_feature.ImagesFeature(
        reader=images_feature.get_default_reader(test_utils.mock_video_path()),
        read_by=images_feature.READ_BY_FRAME_NUMBER,
        read_at=[0],
    )
    self.assertEqual(feature1, feature2)

  def test_images_feature_not_equal(self):
    feature1 = images_feature.ImagesFeature(
        reader=images_feature.get_default_reader(test_utils.mock_video_path()),
        read_by=images_feature.READ_BY_FRAME_NUMBER,
        read_at=[0],
    )
    feature2 = images_feature.ImagesFeature(
        reader=images_feature.get_default_reader(test_utils.mock_video_path()),
        read_by=images_feature.READ_BY_TIMESTAMP_MILLIS,
        read_at=[0],
    )
    self.assertNotEqual(feature1, feature2)

  def test_images_feature_not_equal_with_different_type(self):
    self.assertNotEqual(new_feature((1, 1), [0, 1, 2]), 3.14)

  def test_hash_is_deterministic(self):
    feature = images_feature.ImagesFeature(
        reader=video_io.VideoFileReader('/tmp/test.mp4', check_path=False),
        read_by=images_feature.READ_BY_FRAME_NUMBER,
        read_at=[0, 1, 2],
        image_size=(224, 224),
    )
    self.assertEqual(hash(feature), 2246803685166274479)
