"""dataset_creator wraps video sampling operations to generate a Dataset.

The basis for generating your dataset is providing a generator of Examples.
Each Example is basically an OrderedDict whose values are either "simple" values
(booleans, ints, etc..) or "CustomFeatures".

Usage examples:
  - Create an example bank:
    creator = DatasetCreator('parkland', func_which_returns_a_generator)
    creator.create_example_bank()

  - Get the last dataset created for a project:
    creator = DatasetCreator.load_latest_dataset('cholec80')

  - Get a DataFrame with all datasets metadata of some project:
    datasets_df = get_registered_datasets('cholec80')

  - Get a tf.data.Dataset:
    ds = creator.get_tf_dataset()
    * Add augmentations and batching *

  - Get a torch.utils.data.Dataset:
    ds = creator.get_torch_dataset()
    * Add augmentations and batching *

  - Reproduce an example bank containing previously-generated Examples:
    # Find the timestamp you want using the following:
    dataset_creator.get_registered_datasets(<dataset_name>)
    prev_timestamp = datetime.fromisoformat(timestamp_from_table)
    creator = DatasetCreator(<dataset_name>, creation_time=prev_timestamp)
    creator.create_example_bank()
"""

from __future__ import annotations

import collections
import datetime
import functools
import itertools
from multiprocessing import pool as multiprocessing_pool
import os
import queue
import threading
import time
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence

import apache_beam as beam
from apache_beam.options import pipeline_options
from google.cloud import pubsub_v1  # type: ignore[attr-defined]
import more_itertools
import pandas as pd
import tensorflow as tf
import torch  # type: ignore[import]
import tqdm

from dataset_creator import example_bank_shard_dataset
from dataset_creator import example_lib
from dataset_creator import generated_dataset
from dataset_creator import helpers
from dataset_creator.features import images_feature
from dataset_creator.pipeline import dataflow_utils
from dataset_creator.pipeline import pipeline_utils

Example = example_lib.Example
GeneratedDataset = generated_dataset.GeneratedDataset

CURRENT_VERSION = '2.2.1'

# Support endless generators by truncating to "only" 50,000,000 examples.
MAX_EXAMPLES = 50000000

_BATCH_SIZE_FOR_TF_EXAMPLE_PARSING = 32
_TorchMappingDecoder = Callable[[dict[str, Any]], dict[str, Any]]
_ALL_TORCH_DECODERS: Sequence[_TorchMappingDecoder] = [
    images_feature.torch_decode_images
]
_TfMappingDecoder = Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]]
_ALL_TF_DECODERS: Sequence[_TfMappingDecoder] = [
    images_feature.tf_decode_images
]

# =================== Constants for dynamic dataset creation ===================

_MAX_WAITING_EXAMPLES = 20000
_MAX_WAITING_BYTES = 2 * 1024 ** 3
_EXAMPLE_PUBLISHER_THREAD_NAME = '_EXAMPLE_PUBLISHER_THREAD'
_TIMEOUT_BEFORE_ANY_EXAMPLE = 900
_TIMEOUT_AFTER_EXAMPLE = 30
_VCPUS_TO_ADD_ON_EPOCH_START = 1200
_NUM_WORKERS_TO_ADD_ON_EPOCH_START = int(
    _VCPUS_TO_ADD_ON_EPOCH_START / dataflow_utils.NUM_VCPUS_PER_WORKER
)


def _start_in_worker_thread(worker_name: str) -> Any:
  """Returns a decorator that starts an instance method in a worker thread.

  Args:
    worker_name: The name of the worker thread.

  Raises:
    RuntimeError: In case a worker with that name is already running.
  """
  def instance_method_decorator(method) -> Callable[..., None]:
    def maybe_start_threads(*args, **kwargs):
      threading.Thread(
          target=method, args=args, kwargs=kwargs, name=worker_name
      ).start()
    return maybe_start_threads
  return instance_method_decorator


def _tf_element_to_np_element(element: dict[str, tf.Tensor]) -> dict[str, Any]:
  return {k: v.numpy() for k, v in element.items()}


class _IterableTorchDatasetFromTfDataset(torch.utils.data.IterableDataset):
  def __init__(self, tf_dataset: tf.data.Dataset):
    self._tf_dataset = tf_dataset

  def __iter__(self):
    return map(_tf_element_to_np_element, self._tf_dataset)


class _ShardDataset(torch.utils.data.Dataset):
  """A PyTorch Dataset from a single Parquet file."""

  def __init__(self, filename: str, io_spec: dict[str, Any]):
    self._dataset = example_bank_shard_dataset.get_torch_dataset(filename)
    self._io_spec = io_spec

  def __len__(self) -> int:
    return len(self._dataset)

  def __getitem__(self, index: int) -> dict[str, Any]:
    serialized = tf.constant(self._dataset[index])
    parsed_tf_example = tf.io.parse_single_example(serialized, self._io_spec)
    element = example_lib.normalize_parsed_tf_example(parsed_tf_example)
    np_element = _tf_element_to_np_element(element)
    for decoder in _ALL_TORCH_DECODERS:
      np_element = decoder(np_element)
    return np_element


class DatasetCreator:
  """A class which allows for the creation of datasets.

  Attributes:
    dataset_name: A string identifier of the dataset to be / that was previously
      created.
    creation_time: A datetime.datetime object which identifies the creation time
      of the dataset. An identical value of dataset_name might have multiple
      creation_times, but a (dataset_name, creation_time) is a unique identifier
      of the dataset.
    get_generator_cb: A user-defined callback to retrieve a generator. The
      obtained generator should yield unpopulated Example-s.
    max_examples: Slice the generator after this number of examples.
    version: The DatasetCreator version of this creator. In case of a loaded
      creator, this value will be assigned the value used to create the creator
      back when it was first created.
  """

  def __init__(
      self,
      dataset_name: str,
      get_generator: Optional[helpers.Supplier[Iterable[Example]]] = None,
      *,
      creation_time: Optional[datetime.datetime] = None,
      max_examples: int = MAX_EXAMPLES
  ):
    """A creator of datasets and example banks.

    Args:
      dataset_name: A string identifier of the dataset to be / that was
        previously created. The name can only include numbers, english letters
        and the chars {'.', '-', '_'} and it should be short.
      get_generator: A user-defined callback to retrieve a generator. The
        obtained generator should yield unpopulated Example-s.
      creation_time: A datetime.datetime object which identifies the creation
        time of the dataset. An identical value of dataset_name might have
        multiple creation_times, but a (dataset_name, creation_time) is a unique
        identifier of the dataset. If passed as None, the value
        datetime.datetime.now() is used.
      max_examples: Slice the generator after this number of examples.
    """

    if get_generator is None and creation_time is None:
      raise ValueError('Either get_generator / creation_time MUST be specified')
    if get_generator is not None and creation_time is not None:
      raise ValueError(
          'In case of an old dataset, get_generator is filled automatically'
      )
    dataflow_utils.validate_job_name(
        dataflow_utils.get_batch_job_name(dataset_name)
    )

    self.dataset_name = dataset_name

    version = CURRENT_VERSION
    if get_generator is None:
      assert creation_time is not None
      saved_dataset = GeneratedDataset.from_key(dataset_name, creation_time)
      if saved_dataset is None:
        raise ValueError('Dataset not found', dataset_name, creation_time)
      version = saved_dataset.dataset_creator_version or CURRENT_VERSION
      get_generator = saved_dataset.get_examples_generator

    self.version = version
    self.get_generator_cb = get_generator

    if creation_time is None:
      creation_time = datetime.datetime.now(tz=datetime.timezone.utc)

    self.creation_time = creation_time
    self.max_examples = max_examples

    # Members for dynamic dataset creation:
    self._num_written_to_pubsub = 0
    self._num_read_from_pubsub = 0
    self._subscription_id = f'dynamic-dataset-{time.time()}'
    self._keep_writing_to_pubsub = False

  def _get_examples(self) -> Iterable[Example]:
    return itertools.islice(self.get_generator_cb(), self.max_examples)

  def get_dataset_metadata(
      self,
      bucketizer: Optional[Callable[[Example], Sequence[Hashable]]] = None
  ) -> dict[Hashable, int]:
    """Returns statistics of the different distributions of the dataset.

    Note that in the case of multiple labels associated with the same example,
    that example would be counted multiple times.
    Args:
      bucketizer: A function which receives an example and returns a list of all
        (hashable) buckets that example belongs to. Default behavior is to yield
        all values associated with a metadata key which contains 'label' as a
        substring in the example.
    """

    template = more_itertools.first(self._get_examples())

    def default_bucketizer(example: Example) -> list[str]:
      buckets = [f'{self.dataset_name} examples']
      for key, value in example.items():
        if 'label' in key:
          if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            buckets.append(value)  # type: ignore[arg-type]
          else:
            buckets += list(value)  # type: ignore[arg-type]
        if not isinstance(value, type(template[key])):
          raise ValueError(
              f'Value {value} for key {key} has type {type(value)} which is not'
              f' of the same type as the template value {type(template[key])}.'
          )
      return buckets

    bucketizer = bucketizer or default_bucketizer

    dataset_metadata: collections.Counter = collections.Counter()
    with multiprocessing_pool.ThreadPool() as pool:
      for values in tqdm.tqdm(pool.imap(bucketizer, self._get_examples())):
        dataset_metadata.update(values)
    return dict(dataset_metadata)

  def _generated_dataset_to_be_saved(self) -> GeneratedDataset:
    return GeneratedDataset(
        dataset_name=self.dataset_name,
        creation_time=self.creation_time,
        dataset_creator_version=self.version,
        get_examples_generator=self._get_examples,
    )

  def create_example_bank(
      self,
      overwrite_dataset: bool = False,
      skip_local_validation: bool = False,
      additional_pip_requirements: Sequence[str] = (),
      region: str = dataflow_utils.DEFAULT_DATAFLOW_REGION,
      machine_type: str = dataflow_utils.DEFAULT_BATCH_MACHINE_TYPE,
  ) -> None:
    """Generates an on-disk table which contains all sampled examples provided.

    The created table will be placed inside the following root directory:
    /gcs/<project-id>-dataset-creator/datasets
    (Probably) sharded in a directory named <dataset_name>/<creation_timestamp>.

    Args:
      overwrite_dataset: If True, the dataset will always be saved to the DB.
        If False, it will be saved only in case it's not already there.
      skip_local_validation: If True, the pipeline will be started on borg
        without validating locally that a single example goes through the
        pipeline successfully. This is useful when even a single example
        requires a lot of processing, so locally processing it takes a lot of
        time. Default is False.
      additional_pip_requirements: Additional dependencies to be installed using
        pip to the pipeline environment. This might be needed only if you use
        LambdaFeature or GenericLambdaFeature with non-trivial libraries.
      region: Region to run the batch job in. Default is
        dataflow_utils.DEFAULT_DATAFLOW_REGION.
      machine_type: The machine type to use for the batch job. Default is
        dataflow_utils.DEFAULT_BATCH_MACHINE_TYPE.
    """
    if not skip_local_validation:
      # First, before saving or doing anything else - validate that the pipeline
      # succeeds on one example.
      self.validate_pipeline()
    self.save(overwrite=overwrite_dataset)
    dataflow_utils.run_batch(
        self.dataset_name,
        self.creation_time,
        output_path=self.get_example_bank_prefix(),
        additional_pip_requirements=additional_pip_requirements,
        region=region,
        machine_type=machine_type,
    )

  def validate_pipeline(self, validate_io_spec: bool = True):
    """Validates the pipeline can process an example from the generator.

    Args:
      validate_io_spec: If true, validates that the processed example contains
        the keys expected from the template.

    Raises:
      ValueError: In any case the pipeline fails to process the first example.
    """
    saved_dataset = self._generated_dataset_to_be_saved()
    first_example = more_itertools.first(self._get_examples())
    input_queue = collections.deque([first_example])
    output_queue: collections.deque[Example] = collections.deque([])

    validator_pipeline = pipeline_utils.get_direct_streaming_pipeline(
        saved_dataset,
        input_queue,
        output_queue,
        num_examples=1,
        skip_faulty_examples=False,
    )

    options = pipeline_options.PipelineOptions(flags=[], streaming=True)
    runner = beam.runners.DirectRunner()
    try:
      runner.run(validator_pipeline, options=options).wait_until_finish()
    except Exception as e:
      raise ValueError('Pipeline failed.') from e

    assert output_queue, 'First example processing failed. More info in logs.'

    if validate_io_spec:
      io_spec = first_example.get_finalized_tf_example_io_spec()
      output_features = output_queue.popleft().to_tf_example().features.feature
      missing_keys = list(set(io_spec.keys()) - set(output_features.keys()))
      extra_keys = list(set(output_features.keys()) - set(io_spec.keys()))
      if missing_keys or extra_keys:
        raise ValueError(
            f'Processed example has missing keys {missing_keys}, and extra '
            f'keys: {extra_keys}.'
        )

  def get_example_bank_prefix(self) -> str:
    """Returns the path to the project's creator directory."""
    return os.path.join(
        helpers.get_project_bucket_path(),
        'datasets',
        self.dataset_name,
        self.creation_time.strftime('%Y%m%d_%H%M%S'),
        'sampled-table'
    )

  def _validate_example_bank(self) -> Sequence[str]:
    if self.version != CURRENT_VERSION:
      raise ValueError('Example bank must be rebuilt, version is too old.')
    example_bank_filenames = helpers.glob_prefix(self.get_example_bank_prefix())
    if not example_bank_filenames:
      raise ValueError('Example bank does not exist.')
    return example_bank_filenames

  def get_torch_dataset(self) -> torch.utils.data.Dataset:
    """Returns a torch dataset identical to that of get_tf_dataset."""
    if self._is_example_bank_valid():
      return self.get_static_torch_dataset()
    return self.get_dynamic_torch_dataset()

  def get_static_torch_dataset(
      self, example_bank_files: Sequence[str] = (),
  ) -> torch.utils.data.Dataset:
    """Returns a torch dataset identical to that of get_static_tf_dataset.

    Args:
      example_bank_files: Optional. If given, uses the given filenames as
        example bank files.
    """
    tf.config.experimental.set_visible_devices([], 'GPU')
    template = more_itertools.first(self._get_examples())
    io_spec = template.get_finalized_tf_example_io_spec()
    example_bank_files = example_bank_files or self._validate_example_bank()
    return torch.utils.data.ConcatDataset(
        [_ShardDataset(fn, io_spec) for fn in example_bank_files]
    )

  def get_dynamic_torch_dataset(self) -> torch.utils.data.Dataset:
    """Returns a torch dataset identical to that of get_dynamic_tf_dataset."""
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.run_functions_eagerly(True)
    return _IterableTorchDatasetFromTfDataset(self.get_dynamic_tf_dataset())

  def get_tf_dataset(self) -> tf.data.Dataset:
    """Returns a tf.data.Dataset populated with all features.

    Returns:
      A tf.data.Dataset. Elements in the returned dataset keys correspond to
      keys in the user generator:
        1. If the value corresponding to some key was "simple" (bool, list[int],
           etc, ...), that key will be one of the keys of the tf.data.Dataset.
        2. If the value corresponding to some key was a CustomFeature, all keys
           corresponding to that entry in the resulting tf.data.Dataset will
           be nested keys (see: example_lib.nested_key), whose feature_name is
           key.
    """
    if self._is_example_bank_valid():
      return self.get_static_tf_dataset()
    return self.get_dynamic_tf_dataset()

  def _is_example_bank_valid(self) -> bool:
    return bool(helpers.glob_prefix(self.get_example_bank_prefix()))

  @_start_in_worker_thread(_EXAMPLE_PUBLISHER_THREAD_NAME)
  def _push_all_examples_to_pubsub(self) -> None:
    dataset_id = f'{self.dataset_name}_{self.creation_time}'
    template = more_itertools.first(self._get_examples())
    num_features = template.num_custom_features
    attributes = {
        pipeline_utils.DATASET_ID_PUBSUB_ATTRIBUTE: dataset_id,
        pipeline_utils.SUBSCRIPTION_ATTRIBUTE: self._subscription_id,
        pipeline_utils.NUM_FEATURES_ATTRIBUTE: str(num_features),
    }

    condition = threading.Condition()

    def wait_predicate() -> bool:
      num_waiting = self._num_written_to_pubsub - self._num_read_from_pubsub
      return num_waiting < _MAX_WAITING_EXAMPLES

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(
        helpers.get_project_id(), pipeline_utils.UNPOPULATED_TOPIC
    )
    self._keep_writing_to_pubsub = True
    for example in self._get_examples():
      if not self._keep_writing_to_pubsub:
        break
      with condition:
        condition.wait_for(wait_predicate, timeout=1)
      publisher.publish(topic_path, data=example.to_bytes(), **attributes)
      self._num_written_to_pubsub += 1
    self._keep_writing_to_pubsub = False

  def _get_serialized_examples_with_beam(
      self, num_features: int
  ) -> Iterable[bytes]:
    """Yields serialized populated Examples."""
    # Start by creating the output subscription, to avoid race conditions.
    filters = {
        pipeline_utils.SUBSCRIPTION_ATTRIBUTE: self._subscription_id
    }
    subscription_path = helpers.get_pubsub_subscription(
        pipeline_utils.POPULATED_TOPIC, self._subscription_id, **filters
    )

    output_queue: queue.Queue[bytes] = queue.Queue()

    def pull_message_from_pubsub(message):
      attribute = message.attributes[pipeline_utils.SUBSCRIPTION_ATTRIBUTE]
      if attribute == self._subscription_id:
        output_queue.put(message.data)
        self._num_read_from_pubsub += 1
      message.ack()

    with pubsub_v1.SubscriberClient() as subscriber:
      subscriber.subscribe(
          subscription_path,
          flow_control=pubsub_v1.types.FlowControl(
              max_bytes=_MAX_WAITING_BYTES, max_messages=_MAX_WAITING_EXAMPLES,
          ),
          callback=pull_message_from_pubsub,
      )

      self._num_written_to_pubsub = 0
      self._num_read_from_pubsub = 0
      added_workers = 0

      try:
        # Schedule returning to the default number of min_num_workers for the
        # streaming pipeline, as the finally statement might not always run in
        # all use cases, so we have a fallback.
        dataflow_utils.schedule_returning_to_default_min_workers(num_features)
        added_workers = dataflow_utils.update_job_min_workers(
            num_features,
            num_workers_to_add=_NUM_WORKERS_TO_ADD_ON_EPOCH_START,
        )
        self._push_all_examples_to_pubsub()

        timeout = _TIMEOUT_BEFORE_ANY_EXAMPLE
        while True:
          try:
            yield output_queue.get(block=True, timeout=timeout)
            timeout = _TIMEOUT_AFTER_EXAMPLE
          except (queue.Empty, KeyboardInterrupt) as e:
            if (
                isinstance(e, KeyboardInterrupt) or
                (not self._keep_writing_to_pubsub)
            ):
              break
      finally:
        self._keep_writing_to_pubsub = False
        dataflow_utils.update_job_min_workers(
            num_features, num_workers_to_add=-added_workers
        )

  def get_dynamic_tf_dataset(self) -> tf.data.Dataset:
    """Returns a tf.data.Dataset from Examples generated dynamically.

    Please note that dynamic dataset creation cannot run in parallel for
    different DatasetCreator instances, and that if some Examples share objects,
    these objects MUST be thread-safe.

    Raises:
      RuntimeError: In case the streaming pipeline is not running.
    """
    template = more_itertools.first(self._get_examples())
    num_custom_features = template.num_custom_features
    if not dataflow_utils.is_streaming_job_running(num_custom_features):
      raise RuntimeError(
          'The streaming pipeline that populates Examples with '
          f'{num_custom_features} CustomFeatures is not running. Please run it '
          f'by running dataflow_utils.run_streaming({num_custom_features}), '
          'then waiting for ~15 minutes for the pipeline to start.'
      )
    self.validate_pipeline(validate_io_spec=False)
    dataset = tf.data.Dataset.from_generator(
        lambda: self._get_serialized_examples_with_beam(num_custom_features),
        output_signature=tf.TensorSpec([], tf.string)
    )
    return self._decode_tf_dataset(dataset)

  def get_static_tf_dataset(
      self, example_bank_files: Sequence[str] = ()
  ) -> tf.data.Dataset:
    """Returns a tf.data.Dataset sampled from example bank.

    Args:
      example_bank_files: Optional. If given, uses the given filenames as
        example bank files.
    """

    example_bank_files = example_bank_files or self._validate_example_bank()

    dataset = tf.data.Dataset.from_tensor_slices(example_bank_files)
    dataset = dataset.interleave(
        example_bank_shard_dataset.get_tf_dataset,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return self._decode_tf_dataset(dataset)

  def _decode_tf_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Decodes the given dataset.

    Args:
      dataset: The dataset to be decoded. All elements in the dataset are
        assumed to be serialized tf examples, whose specs are those given by the
        example_lib.Example template.

    Returns:
      The decoded dataset.
    """
    template = more_itertools.first(self._get_examples())
    io_spec = template.get_finalized_tf_example_io_spec()
    dataset = dataset.batch(
        _BATCH_SIZE_FOR_TF_EXAMPLE_PARSING, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        functools.partial(tf.io.parse_example, features=io_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.unbatch()
    dataset = dataset.map(
        example_lib.normalize_parsed_tf_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    for decoder in _ALL_TF_DECODERS:
      dataset = dataset.map(decoder, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

  @classmethod
  def load_latest_dataset(cls, dataset_name: str) -> DatasetCreator:
    """Returns a DatasetCreator object corresponding to the latest dataset_name.

    Args:
      dataset_name: The dataset_name to retrieve.

    Raises:
      ValueError: In case no datasets with name dataset_name were found.
    """
    registered_datasets = get_registered_datasets(dataset_name)
    if registered_datasets.empty:
      raise ValueError(f'No datasets with name {dataset_name} found!')
    latest_creation_time = registered_datasets.iloc[-1]['creation_time']
    return cls(
        dataset_name,
        creation_time=latest_creation_time.to_pydatetime(),
    )

  def save(self, overwrite: bool = False) -> None:
    """Saves this DatasetCreator instance to the DB for reproducibility.

    Please note that saving the generated Examples might take some time,
    depending on the Examples generator pace.

    Args:
      overwrite: If True, writes this dataset even if it already exists in the
        DB.
    """
    is_saved = GeneratedDataset.from_key(self.dataset_name, self.creation_time)
    if is_saved and not overwrite:
      return
    self._generated_dataset_to_be_saved().save()


def get_registered_datasets(dataset_name: str) -> pd.DataFrame:
  """Returns a pd.DataFrame containing datasets sorted by creation_time.

  Args:
    dataset_name: The dataset_name to retrieve.
  """
  rows = generated_dataset.get_registered_datasets(dataset_name)
  rows_with_prefix = []
  for _, creation_time in rows:
    creator = DatasetCreator(dataset_name, creation_time=creation_time)
    prefix = creator.get_example_bank_prefix()
    if not helpers.glob_prefix(prefix):
      prefix = ''
    rows_with_prefix.append((dataset_name, creation_time, prefix))
  df = pd.DataFrame(
      data=rows_with_prefix,
      columns=['dataset_name', 'creation_time', 'example_bank_prefix']
  )
  return df.sort_values(by=['creation_time'])
