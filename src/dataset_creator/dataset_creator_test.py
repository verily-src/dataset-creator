"""Tests for dataset_creator."""

from datetime import datetime
import functools
import io
import os
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union
from unittest import mock

from absl.testing import parameterized  # type: ignore[import]
from google.cloud import spanner  # type: ignore[attr-defined]
import more_itertools
import numpy as np
import pytest
import tensorflow as tf
import torch  # type: ignore[import]
from typing_extensions import TypeAlias

from dataset_creator import dataset_creator
from dataset_creator import example_lib
from dataset_creator import helpers
from dataset_creator import test_utils
from dataset_creator.features import base_feature
from dataset_creator.features import fields
from dataset_creator.pipeline import dataflow_utils

# pylint: disable=protected-access

_nested_key = example_lib.nested_key
_TfDataset: TypeAlias = tf.data.Dataset
_TorchDataset: TypeAlias = torch.utils.data.Dataset


@pytest.fixture(scope='function')
def _emulate_tf_dataset_from_generator():
  """This fixture is used for coverage reasones.

  This is because tf.data is not covered in coverage analysis.
  """
  def from_generator(*args, **kwargs):
    generator = kwargs.get('generator', args[0])
    values = list(generator())
    return tf.data.Dataset.from_tensor_slices(values)

  original_from_generator = tf.data.Dataset.from_generator
  tf.data.Dataset.from_generator = from_generator
  yield
  tf.data.Dataset.from_generator = original_from_generator


@pytest.fixture(scope='function')
def _emulate_dataflow_streaming_job():
  original_is_streaming_job_running = dataflow_utils.is_streaming_job_running
  original_update_job = dataflow_utils.update_job_min_workers
  original_schedule = dataflow_utils.schedule_returning_to_default_min_workers

  dataflow_utils.is_streaming_job_running = lambda *_: True
  dataflow_utils.update_job_min_workers = lambda *_, **_2: 0
  dataflow_utils.schedule_returning_to_default_min_workers = lambda *args: None

  yield
  dataflow_utils.is_streaming_job_running = original_is_streaming_job_running
  dataflow_utils.update_job_min_workers = original_update_job
  dataflow_utils.schedule_returning_to_default_min_workers = original_schedule


def save_creators_to_spanner(creators: list[dataset_creator.DatasetCreator]):
  for creator in creators:
    creator._generated_dataset_to_be_saved().save()


class ErrorProneFeature(base_feature.CustomFeature):
  """The following feature is error prone since it accepts a float.

  The proper way to handle a float is to force it to be a float in __init__.
  Otherwise, an int is a valid argument, but is not equivalent in terms of tf
  features.
  """

  def __init__(self, value: float, **kwargs):
    super().__init__(value=value, **kwargs)
    self._value = value

  def split(self) -> Sequence[float]:
    return [self._value]

  def process(self, metadata_value: float, _) -> float:
    return metadata_value

  def merge(self, values: Sequence[float]) -> Mapping[str, float]:
    return {'value': values[0]}

  @property
  def output_signature(self) -> Mapping[str, type[base_feature.ValueFeature]]:
    return {'value': float}


def setUpModule():
  example_lib._TYPE_NAME_TO_TYPE['ErrorProneFeature'] = ErrorProneFeature
  dataflow_utils.run_streaming(1)


class DatasetCreatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.db = helpers.get_db()
    with self.db.batch() as batch:
      batch.delete('GeneratedDatasets', keyset=spanner.KeySet(all_=True))

    self.dataset_name = 'dataset-creator-test'
    self.creator = dataset_creator.DatasetCreator(
        self.dataset_name, test_utils.get_examples_generator,
    )

  @parameterized.named_parameters(
      (
          'conflicting_params',
          test_utils.get_examples_generator,
          datetime.now(),
          'In case of an old dataset, get_generator is filled automatically',
      ),
      (
          'insufficient_params',
          None,
          None,
          'Either get_generator / creation_time MUST be specified'
      ),
      (
          'creation_time_not_corresponding_to_a_dataset',
          None,
          datetime.now(),
          'Dataset not found'
      ),
  )
  def test_instantiation_raises(
      self,
      get_generator: Optional[Callable[[], Iterable[dataset_creator.Example]]],
      creation_time: Optional[datetime],
      error_message: str,
  ):
    with self.assertRaisesRegex(ValueError, error_message):
      dataset_creator.DatasetCreator(
          'test-dataset', get_generator, creation_time=creation_time
      )

  def test_get_dataset_metadata(self):
    self.assertEqual(
        self.creator.get_dataset_metadata(),
        {f'{self.dataset_name} examples': 5, 0: 3, 1: 1, 2: 1}
    )

  def test_get_dataset_metadata_raises_example_type_doesnt_match_template(self):
    type_mismatching_examples = [
        example_lib.Example({'test': 1}), example_lib.Example({'test': 2.0})
    ]
    creator = dataset_creator.DatasetCreator(
        'test', lambda: type_mismatching_examples
    )
    with self.assertRaisesRegex(ValueError, 'not of the same type'):
      creator.get_dataset_metadata()

  def test_get_dataset_metadata_with_non_iterable_labels(self):
    def get_examples_with_primitive_labels():
      label_key = test_utils.LABELS_FEATURE_NAME
      for example in self.creator.get_generator_cb():
        label_feature = {label_key: example[label_key][0]}
        yield example_lib.Example(example | label_feature)

    creator = dataset_creator.DatasetCreator(
        'test', get_examples_with_primitive_labels
    )
    self.assertEqual(
        creator.get_dataset_metadata(),
        {'test examples': 5, 0: 3, 1: 1, 2: 1}
    )

  def _get_creator_and_dataset(
      self,
      *,
      include_model_outputs: bool = False,
      resolution: tuple[int, int] = (224, 224),
      include_frames: bool = True,
      is_dynamic: bool = False,
      is_torch: bool = False,
  ) -> tuple[dataset_creator.DatasetCreator, Union[_TfDataset, _TorchDataset]]:
    """Returns the DatasetCreator and dataset as requested.

    The default behavior returns the static Tensorflow dataset for a creator
    that includes only (224, 224) frames.
    """
    generator_kwargs: dict[str, Any] = {'image_size': resolution}
    if include_model_outputs:
      model_path, layers, example_to_inputs = test_utils.inference_parameters()
      generator_kwargs['keras_model_path'] = model_path
      generator_kwargs['outputs_layer_names'] = layers
      generator_kwargs['example_to_inputs'] = example_to_inputs
    if not include_frames:
      generator_kwargs['inference_only'] = True
    get_generator = functools.partial(
        test_utils.get_examples_generator, **generator_kwargs
    )
    creator = dataset_creator.DatasetCreator(self.dataset_name, get_generator)

    if not is_dynamic:
      creator.get_example_bank_prefix = (  # type: ignore[method-assign]
          test_utils.mock_example_bank_prefix
      )

    if is_torch:
      dataset: Union[_TorchDataset, _TfDataset] = creator.get_torch_dataset()
    else:
      dataset = creator.get_tf_dataset()
    return creator, dataset

  @pytest.mark.usefixtures('_emulate_dataflow_streaming_job')
  def test_dataset_with_no_frames(self):
    _, dataset = self._get_creator_and_dataset(
        include_frames=False, is_dynamic=True,
    )
    images_key = _nested_key(test_utils.IMAGES_FEATURE_NAME, fields.IMAGES)
    tensors = more_itertools.first(dataset)
    self.assertIsNotNone(tensors)
    self.assertNotIn(images_key, tensors)

  @parameterized.named_parameters(
      ('with_model_outputs', True, False),
      ('without_model_outputs', False, False),
      ('dynamic_dataset', False, True),
  )
  @pytest.mark.usefixtures('_emulate_dataflow_streaming_job')
  def test_get_tf_dataset(self, include_model_outputs: bool, is_dynamic: bool):
    resolution = (112, 224)
    creator, dataset = self._get_creator_and_dataset(
        include_model_outputs=include_model_outputs,
        resolution=resolution,
        is_dynamic=is_dynamic
    )
    self.assertDatasetMatchesCreator(
        creator, list(dataset), resolution  # type: ignore[arg-type]
    )

  @pytest.mark.usefixtures(
      '_emulate_tf_dataset_from_generator', '_emulate_dataflow_streaming_job'
  )
  def test_get_tf_dataset_for_coverage_purposes(self):
    resolution = (112, 224)
    creator, dataset = self._get_creator_and_dataset(
        resolution=resolution, is_dynamic=True
    )
    self.assertDatasetMatchesCreator(creator, list(dataset), resolution)

  def test_get_dynamic_tf_dataset_raises_when_pipeline_not_running(self):
    # Note that we are not emulating the dataflow streaming job for this test.
    with self.assertRaisesRegex(RuntimeError, 'pipeline.*not running'):
      self.creator.get_dynamic_tf_dataset()

  @pytest.mark.usefixtures('_emulate_dataflow_streaming_job')
  def test_dynamic_dataset_stops_publishing_to_pubsub_upon_exit(self):
    # Patch so we don't push examples after the first one. This way we can
    # simulate a GeneratorExit before the dataset finishes publishing
    with mock.patch(
        'dataset_creator.dataset_creator._MAX_WAITING_EXAMPLES',
        1
    ):
      with mock.patch(
          'dataset_creator.dataset_creator._BATCH_SIZE_FOR_TF_EXAMPLE_PARSING', 1  # pylint: disable=line-too-long
      ):
        _, dataset = self._get_creator_and_dataset(is_dynamic=True)
        for _ in dataset:
          self.assertIn(
              dataset_creator._EXAMPLE_PUBLISHER_THREAD_NAME,
              map(lambda thread: thread.name, threading.enumerate())
          )
          break
        time.sleep(3)  # Let the publisher thread some time to stop
        self.assertNotIn(
            dataset_creator._EXAMPLE_PUBLISHER_THREAD_NAME,
            map(lambda thread: thread.name, threading.enumerate())
        )

  def test_dataset_contains_exactly_the_expected_features(self):
    _, dataset = self._get_creator_and_dataset()
    expected_keys = [
        fields.IMAGES_READ_AT,
        fields.IMAGES_READ_BY,
        fields.IMAGES_PATH,
        fields.IMAGES,
        fields.IMAGES_NUM_BITS,
        fields.IMAGES_FPS,
    ]
    expected_features = [test_utils.LABELS_FEATURE_NAME] + [
        _nested_key(test_utils.IMAGES_FEATURE_NAME, key)
        for key in expected_keys
    ]
    for tensors in dataset:
      self.assertSameElements(expected_features, tensors.keys())

  def assertDatasetMatchesCreator(
      self,
      creator: dataset_creator.DatasetCreator,
      dataset: Sequence[Mapping[str, Any]],
      expected_resolution: tuple[int, int],
  ):
    examples = list(creator._get_examples())
    template = examples[0]

    expected_labels = [
        example[test_utils.LABELS_FEATURE_NAME] for example in examples
    ]
    actual_labels = [
        tensors[test_utils.LABELS_FEATURE_NAME].numpy() for tensors in dataset
    ]
    self.assertCountEqual(expected_labels, actual_labels)

    images_key = test_utils.IMAGES_FEATURE_NAME
    if images_key in template:
      expected_timestamps = [
          example[images_key]._read_at  # type: ignore[union-attr]
          for example in examples
      ]
      timestamps_key = _nested_key(images_key, fields.IMAGES_READ_AT)
      actual_timestamps = [
          tensors[timestamps_key].numpy() for tensors in dataset
      ]
      self.assertCountEqual(expected_timestamps, actual_timestamps)

      images_key = _nested_key(images_key, fields.IMAGES)
      self.assertTrue(
          [
              tensors[images_key].numpy().shape == (*expected_resolution, 3)
              for tensors in dataset
          ],
          f'Output image has shape != ({expected_resolution, 3})'
      )

    if test_utils.INFERENCE_FEATURE_NAME in template:
      inference_feature = examples[0][test_utils.INFERENCE_FEATURE_NAME]
      layer_names = (
          inference_feature._outputs_layer_names  # type: ignore[union-attr]
      )
      for tensors in dataset:
        for layer_name in layer_names:
          layer_key = _nested_key(test_utils.INFERENCE_FEATURE_NAME, layer_name)
          self.assertNotEmpty(tensors[layer_key])

  def test_get_latest_dataset(self):
    later_creator = dataset_creator.DatasetCreator(
        self.creator.dataset_name, test_utils.get_examples_generator
    )
    save_creators_to_spanner([later_creator, self.creator])
    creator = dataset_creator.DatasetCreator.load_latest_dataset(
        self.creator.dataset_name
    )
    self.assertEqual(creator.dataset_name, self.dataset_name)
    self.assertEqual(creator.creation_time, later_creator.creation_time)

  def test_load_latest_dataset_raises_with_non_existing_dataset_name(self):
    with self.assertRaisesRegex(ValueError, 'No datasets with name'):
      dataset_creator.DatasetCreator.load_latest_dataset('test-dataset')

  def test_get_registered_datasets(self):
    later_creator = dataset_creator.DatasetCreator(
        self.creator.dataset_name, test_utils.get_examples_generator
    )
    save_creators_to_spanner([later_creator, self.creator])
    datasets_df = dataset_creator.get_registered_datasets(
        self.creator.dataset_name
    )
    self.assertCountEqual(
        datasets_df['creation_time'],  # pylint: disable=unsubscriptable-object
        [self.creator.creation_time, later_creator.creation_time],
    )

  def test_generator_slicing(self):
    creator = dataset_creator.DatasetCreator(
        self.dataset_name, test_utils.get_examples_generator, max_examples=2
    )
    self.assertLen(list(creator._get_examples()), 2)

  def test_dataset_created_with_current_version(self):
    self.assertEqual(self.creator.version, dataset_creator.CURRENT_VERSION)

  def test_create_example_bank_raises_with_stateful_example_to_input(self):
    keras_model_path, outputs_layer_names, _ = test_utils.inference_parameters()

    creator = dataset_creator.DatasetCreator(
        self.dataset_name,
        functools.partial(
            test_utils.get_examples_generator,
            keras_model_path=keras_model_path,
            outputs_layer_names=outputs_layer_names,
            example_to_inputs=lambda _: np.range(3),
        )
    )
    with self.assertRaises(ValueError):
      creator.create_example_bank()

  def test_dataset_not_raises_with_valid_model_parameters(self):
    keras_model_path, outputs_layer_names, example_to_input = (
        test_utils.inference_parameters(input_size=224 * 224 * 3)
    )
    creator = dataset_creator.DatasetCreator(
        self.dataset_name,
        functools.partial(
            test_utils.get_examples_generator,
            image_size=(224, 224),
            keras_model_path=keras_model_path,
            outputs_layer_names=outputs_layer_names,
            example_to_inputs=example_to_input,
        )
    )
    creator.validate_pipeline()

  def test_validate_pipeline_raises_with_bad_model_inputs(self):
    keras_model_path, outputs_layer_names, example_to_input = (
        test_utils.inference_parameters(input_size=10)
    )
    creator = dataset_creator.DatasetCreator(
        self.dataset_name,
        functools.partial(
            test_utils.get_examples_generator,
            keras_model_path=keras_model_path,
            outputs_layer_names=outputs_layer_names,
            example_to_inputs=example_to_input,
        )
    )
    with self.assertRaisesRegex(ValueError, 'Pipeline failed.'):
      # The extracted images don't match the expected input_size.
      creator.validate_pipeline()

  def test_validate_pipeline_raises_with_a_bad_feature(self):
    bad_feature = ErrorProneFeature(1)  # Note that an int is passed, not float
    bad_example = example_lib.Example({'bad_feature': bad_feature})
    creator = dataset_creator.DatasetCreator('test', lambda: [bad_example])
    with self.assertRaisesRegex(ValueError, 'missing keys'):
      creator.validate_pipeline()

  def assertTorchDatasetMatchesTfDataset(
      self,
      torch_dataset: _TorchDataset,
      tf_dataset: _TfDataset,
      order_by: str,
  ):
    tf_iterator = list(tf_dataset)
    tf_dataset_as_list = []
    torch_dataset_as_list = list(torch_dataset)  # type: ignore[call-overload]
    self.assertLen(tf_iterator, len(torch_dataset_as_list))
    # Ordering of the tensors is not guaranteed to match, so start with ordering
    # according to the given key.
    for torch_tensors in torch_dataset_as_list:
      torch_tensor = torch_tensors[order_by]
      matching_tf_tensor_index = np.argmin(
          [
              np.abs(torch_tensor - tf_tensors[order_by].numpy()).sum()
              for tf_tensors in tf_iterator
          ]
      )
      tf_dataset_as_list.append(tf_iterator[matching_tf_tensor_index])
    for torch_tensors, tf_tensors in zip(
        torch_dataset_as_list, tf_dataset_as_list
    ):
      self.assertSameElements(torch_tensors.keys(), tf_tensors.keys())
      for order_by_key in torch_tensors:
        self.assertAllEqual(
            torch_tensors[order_by_key],
            tf_tensors[order_by_key].numpy()
        )

  @parameterized.named_parameters(
      ('static_dataset', False), ('dynamic_dataset', True),
  )
  @pytest.mark.usefixtures('_emulate_dataflow_streaming_job')
  def test_torch_dataset_matches_tf_dataset(self, is_dynamic: bool):
    _, tf_dataset = self._get_creator_and_dataset(is_dynamic=is_dynamic)
    _, torch_dataset = self._get_creator_and_dataset(
        is_dynamic=is_dynamic, is_torch=True
    )
    self.assertTorchDatasetMatchesTfDataset(
        torch_dataset,  # type: ignore[arg-type]
        tf_dataset,  # type: ignore[arg-type]
        order_by='images/images'
    )

  def test_static_torch_dataset_supports_subprocess_workers(self):
    creator, torch_dataset = self._get_creator_and_dataset(
        is_torch=True, is_dynamic=False
    )
    loader = torch.utils.data.DataLoader(torch_dataset, num_workers=2)
    self.assertLen(loader, len(list(creator.get_generator_cb())))

  def test_get_static_dataset_raises_with_invalid_example_bank_path(self):
    # The default path for an example_bank does not exist while testing.
    with self.assertRaisesRegex(ValueError, 'Example bank does not exist'):
      self.creator.get_static_tf_dataset()

  def test_get_static_dataset_raises_with_old_example_bank_format(self):
    with mock.patch(
        'dataset_creator.dataset_creator.CURRENT_VERSION', 'test'
    ):
      with self.assertRaisesRegex(ValueError, 'version is too old'):
        self.creator.get_static_tf_dataset()

  def test_instantiation_raises_with_invalid_dataset_name(self):
    with self.assertRaisesRegex(ValueError, 'Invalid name!'):
      dataset_creator.DatasetCreator('!', lambda: [])

  @parameterized.named_parameters(('false', False), ('true', True))
  def test_save_overwrite(self, overwrite: bool):
    self.creator.save()
    original_version = self.creator.version
    self.creator.version = 'test_version'
    self.creator.save(overwrite=overwrite)
    reproduced_creator = dataset_creator.DatasetCreator(
        self.creator.dataset_name, creation_time=self.creator.creation_time
    )
    if overwrite:
      expected_version = self.creator.version
    else:
      expected_version = original_version
    self.assertEqual(expected_version, reproduced_creator.version)

  @mock.patch('sys.stdout', new_callable=io.StringIO)
  def test_create_example_bank_prints_link_to_job(self, mock_stdout):
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch.object(
          self.creator,
          'get_example_bank_prefix',
          lambda: os.path.join(temp_dir, 'test')
      ):
        self.creator.create_example_bank()
      self.assertIn('https://', mock_stdout.getvalue())

# pylint: enable=protected-access
