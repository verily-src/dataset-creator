r"""A utility module to help with population of Examples.

Populating might refer to different types of data. For example, one type of
population could be reading frames from videos. Another is adding feature
vectors to the Example's metadata.
Generally speaking, population means "taking the relevant metadata from the
example, and adding the data to it."

The general idea of how we populate each example in the pipeline is:
        example
         //|\\                     ExampleSplitter
   processing inputs
         |||||   <---- context     ContextSetup, ValueProcessorWithContext
   processed outputs
         \\|//                     ValueMerger
        example

To that end, the ExamplePopulator class is exposed publicly.
ExamplePopulator is class containing the required functions:
  ExampleSplitter: Splits an example to multiple values, each is an individual
    processing input.
  ContextSetup: Creates a context for processing.
  ValueProcessorWithContext: Processes a value using some common context, to
    create a new value. This callable is not allowed to perform any mutation
    on the original input. A return value of None indicates a failure in
    processing.
  ValueMerger: Takes an example and a sorted sequence of values (ordered by
    the "correct" order) and merges them to a single example.
"""

import collections
import dataclasses
import logging
import os
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import apache_beam as beam
import more_itertools
from typing_extensions import TypeAlias

from dataset_creator import example_lib
from dataset_creator import helpers
from dataset_creator.features import base_feature

Example = example_lib.Example
CustomFeaturesOnlyExample = Example
PCollection: TypeAlias = beam.PCollection


DatasetClassifier = Callable[[Example], str]
OperationIdAssigner = Callable[[Example], Sequence[base_feature.ValueFeature]]
ExampleSplitter = Callable[[Example], Sequence[Any]]
ContextSetup = Callable[[CustomFeaturesOnlyExample], Any]
ValueProcessorWithContext = Callable[[CustomFeaturesOnlyExample, Any, Any], Any]
ValueMerger = Callable[[Example, Sequence[Any]], Example]


class ExamplePopulator(beam.PTransform):
  r"""Populates a PCollection of examples.

  Generally speaking, population in this context means "taking the relevant
  metadata from the example, and adding the data to it." The Examples in the
  PCollection might be taken from different datasets, which are classified by
  example_classifier. Same-dataset examples are characterized by always having
  the same context as one another.
  """

  def __init__(
      self,
      dataset_classifier: DatasetClassifier,
      operation_id_assigner: OperationIdAssigner,
      splitter: ExampleSplitter,
      context_setup: ContextSetup,
      value_processor: ValueProcessorWithContext,
      value_merger: ValueMerger,
      skip_faulty_examples: bool = True
  ):
    r"""Instantiate a Populator instance.

    The general idea of the pipeline's operation is:
    example   example
       \\      //                      operation_id_assigner
    same op_id examples
            |    \___  \___            take first op_id example
     single_example  \     \
          //|\\       \_    \____      splitter (once per op_id)
    processing inputs   \____    \
          |||||  <-- context |    |    context_setup, value_processor
    processed outputs copies |    |
    \\|// <-\     \\|// <----|    |    value_merger (into all op_id examples)
    example  \   example   ______/
              \___________/

    Args:
      dataset_classifier: Classifies an Example into a string label. Later
        contexts are assumed to be unique per label.
      operation_id_assigner: Provides an id to each example that identifies the
        population operation to be performed by this Populator. This id is used
        to group examples that need to go through the same operation, thus
        saving a lot of duplicated processing.
      splitter: Splits an example to multiple values, each is an individual
        processing input. If the sequence is empty, no processing takes place.
      context_setup: Creates a context for processing. The context is assumed to
        be dependent on the CustomFeatures of the example alone.
      value_processor: Processes a value using some common context, to create a
        new value. A return value of None indicates a failure in processing.
      value_merger: Takes an example and a sorted sequence of values (ordered by
        the "correct" order) and merges them to a single example.
      skip_faulty_examples: If False, an example causing an exception will raise
        the exception, instead of silently dropped. Default is True. PLEASE NOTE
        that setting this value to False may cause the entire pipeline to fail!

    Raises:
      Exception: In any case the given callbacks raise an exception, assuming
        skip_faulty_examples is False.
    """
    super().__init__()
    self._dataset_classifier = dataset_classifier
    self._operation_id_assigner = operation_id_assigner
    self._splitter = splitter
    self._context_setup = context_setup
    self._value_processor = value_processor
    self._value_merger = value_merger
    self._skip_faults = skip_faulty_examples

  def expand(self, input_or_inputs: PCollection) -> PCollection:
    """Returns a PCollection with the examples populated.

    Creates the following counters (label being the PTransform's label):
      total_examples - Examples that started the population process.
      dataset_id_errors - Examples we couldn't get dataset_id for.
      empty_examples - Examples that didn't yield any work units.
      total_work_units - The total number of work units started by this stage.
      processing_work_units - Work units that started processing.
      null_work_units - Work units that required no processing.
      failed_work_units - Work units that failed processing.
      completed_work_units - Work units that successfully completed processing.
      merged_examples - Examples successfully merged from processed values.
      failed_merges - Examples that failed to merge from processed values.

    Args:
      input_or_inputs: The examples PCollection to be populated.

    Raises:
      RuntimeError: In case this PTransform is not supported for this pipeline
        / runner configuration.
    """
    return (
        input_or_inputs
        | 'GroupByIdentifier' >> beam.GroupBy(self._operation_id_assigner)
        | 'Split' >> beam.ParDo(
            _SplitToShards(
                self._dataset_classifier, self._splitter, self._skip_faults,
            )
        )
        | 'Reshuffle' >> beam.Reshuffle()
        | 'AddDoKeys' >> beam.Map(
            lambda shard: (shard.dataset_id, shard)
        ).with_output_types(tuple[str, _ExampleWorkUnitShard])
        | 'Do' >> beam.ParDo(
            _TransformShards(
                self._context_setup, self._value_processor, self._skip_faults,
            )
        )
        | 'AddHashes' >> beam.Map(
            lambda shard: (shard.example_hash, shard)
        ).with_output_types(tuple[str, _ExampleWorkUnitShard])
        | 'GroupByKey' >> beam.GroupByKey()
        | 'OmitKeys' >> beam.Map(
            lambda grouped_shards: grouped_shards[1]
        ).with_output_types(Sequence[_ExampleWorkUnitShard])
        | 'PrepareForMerge' >> beam.FlatMap(_split_to_single_example_work_units)
        | 'ReshuffleBeforeMerge' >> beam.Reshuffle()
        | 'Merge' >> beam.ParDo(
            _MergeShards(self._value_merger, self._skip_faults)
        )
    )


@dataclasses.dataclass(frozen=True)
class _ExampleWorkUnitShard:
  """A shard representing a processing unit."""
  dataset_id: str
  example_hash: str
  # Don't use the examples themselves as members since this class is meant to be
  # serialized and deserialized when passing this work unit between workers
  # along the pipeline execution. This is bad because Example instantiation
  # upon deserialization might take a lot of I/O, and also refer to buckets that
  # might not be mounted in the context of deserialization.
  examples_bytes: list[bytes]
  # This is an important optimization for large examples. We must preserve the
  # CustomFeatures of the example when we process values, since the processing
  # of each CustomFeature is stateful of the feature. We SHOULD NOT, however,
  # retain the entire example, as this possibly leads to duplication of
  # unnecessary data, such as the actual images from previous stages. When a
  # single example contains hundreds of images, this can actually cause the
  # pipeline to OOM, since each image is duplicated every time the example is
  # split. The custom_features_only_bytes allows us to retain the necessary
  # parts of the example only.
  custom_features_only_bytes: bytes
  shard_index: int
  num_shards: int
  work_unit_value: Any


class GcsDependentDoFn(beam.DoFn):
  def setup(self):
    gcs_path = os.path.join(
        helpers.get_project_bucket_path(), 'vm_startup_script.sh'
    )
    wait_for_path(gcs_path)


class _SplitToShards(GcsDependentDoFn):
  """Splits a single example to multiple _ExampleWorkUnitShards."""

  def __init__(
      self,
      example_to_dataset_id: DatasetClassifier,
      example_splitter: ExampleSplitter,
      skip_faults: bool = True,
  ):
    """Instantiate a _SplitToShards instance.

    Args:
      example_splitter: A callable that receives an example as an input, and
        outputs a Sequence of values to be processed. If the sequence is empty,
        no processing takes place.
      skip_faults: If True, faulty examples are silently dropped instead of
        causing an exception to be raised.
    """
    super().__init__()
    self._example_to_dataset_id = example_to_dataset_id
    self._example_splitter = example_splitter
    self._skip_faults = skip_faults

  def process(
      self, element: tuple[Any, Sequence[Example]]
  ) -> Iterable[_ExampleWorkUnitShard]:
    """Splits the example into work units.

    Once completed, _ExampleWorkUnitShards stemming from example will be
    yielded. The shard_index of each shard matches its index within the outputs
    of self._example_splitter(example) and the work_unit_value matches the
    output of the same call.

    Args:
      element: A mapping of population identifier to examples to split.
    """
    _, all_examples = element
    example = more_itertools.first(all_examples)
    beam.metrics.Metrics.counter(self.__class__, 'total_examples').inc()
    try:
      dataset_id = self._example_to_dataset_id(example)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('Caught an exception while labeling an example: %s', e)
      logging.info(traceback.format_exc())
      if not self._skip_faults:
        raise
      logging.info('Example labeling failed. %s', example)
      beam.metrics.Metrics.counter(self.__class__, 'dataset_id_errors').inc()
      return

    try:
      values = self._example_splitter(example)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('Caught an exception while splitting an example: %s', e)
      logging.info(traceback.format_exc())
      if not self._skip_faults:
        raise
      values = []
    if not values:
      logging.info('Example splitting failed. %s', example)
      beam.metrics.Metrics.counter(self.__class__, 'empty_examples').inc()
      return

    example_hash = str(uuid.uuid4())
    custom_features_only = Example(
        {
            k: v for k, v in example.items()
            if isinstance(v, base_feature.CustomFeature)
        }
    )
    serialized_examples = [
        equivalent_example.to_bytes() for equivalent_example in all_examples
    ]
    for i, work_unit_value in enumerate(values):
      yield _ExampleWorkUnitShard(
          dataset_id=dataset_id,
          example_hash=example_hash,
          examples_bytes=serialized_examples if i == 0 else [b''],
          custom_features_only_bytes=custom_features_only.to_bytes(),
          shard_index=i,
          num_shards=len(values),
          work_unit_value=work_unit_value,
      )
      beam.metrics.Metrics.counter(self.__class__, 'total_work_units').inc()


class _TransformShards(GcsDependentDoFn):
  """Converts an _ExampleWorkUnitShard to another _ExampleWorkUnitShard.

  This class does most of the heavylifting of populating some feature in an
  example.
  """

  def __init__(
      self,
      setup_context: ContextSetup,
      value_processor: ValueProcessorWithContext,
      skip_faults: bool = True,
  ):
    """Instantiates a _TransformShards instance.

    Args:
      setup_context: A callable to create a context for the value processing
        while setting up a worker.
      value_processor: A callable that takes the input and the context and
        returns a new value. In case the transformation returns None, it is
        assumed that the processing failed.
      skip_faults: If True, faulty examples are silently dropped instead of
        causing an exception to be raised.
    """
    super().__init__()
    self._value_processor = value_processor
    self._setup_context = setup_context
    self._skip_faults = skip_faults
    self._worker_contexts: dict[str, Any] = {}
    self._worker_contexts_lock: Mapping[str, threading.Lock] = (
        collections.defaultdict(threading.Lock)
    )

  def _get_context(self, dataset_id: str, example_bytes: bytes) -> Any:
    with self._worker_contexts_lock[dataset_id]:
      context = self._worker_contexts.get(
          dataset_id, self._setup_context(Example.from_bytes(example_bytes))
      )
      self._worker_contexts[dataset_id] = context
      return context

  def process(
      self,
      element: tuple[str, _ExampleWorkUnitShard],
  ) -> Iterable[_ExampleWorkUnitShard]:
    """Transforms a work unit according to self._value_processor.

    Returns:
      In case the input value is None, there is no context, and this is an only
      shard of this example, no transformation is applied and the original shard
      is returned. Otherwise, returns None.
    """
    beam.metrics.Metrics.counter(self.__class__, 'processing_work_units').inc()

    dataset_id, shard = element
    context = self._get_context(dataset_id, shard.custom_features_only_bytes)

    work_unit_value = shard.work_unit_value
    # When there is nothing to process, support the flow that skips population.
    if work_unit_value is None and context is None and shard.num_shards == 1:
      beam.metrics.Metrics.counter(self.__class__, 'null_work_units').inc()
      return [shard]

    custom_features_only = Example.from_bytes(shard.custom_features_only_bytes)
    try:
      new_value = self._value_processor(
          custom_features_only, work_unit_value, context
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('Caught an exception while processing an example: %s', e)
      logging.info(traceback.format_exc())
      if not self._skip_faults:
        raise
      new_value = None

    if new_value is None:
      logging.info(
          'Example processing failed with input %s. %s',
          shard.work_unit_value,
          Example.from_bytes(shard.custom_features_only_bytes),
      )
      beam.metrics.Metrics.counter(self.__class__, 'failed_work_units').inc()
      return []

    beam.metrics.Metrics.counter(self.__class__, 'completed_work_units').inc()
    return [dataclasses.replace(shard, work_unit_value=new_value)]


def _split_to_single_example_work_units(
    element: Sequence[_ExampleWorkUnitShard]
) -> Iterable[list[_ExampleWorkUnitShard]]:
  element = list(element)
  element.sort(key=lambda work_unit_shard: work_unit_shard.shard_index)
  for serialized in element[0].examples_bytes:
    work_units = []
    for i, shard in enumerate(element):
      work_units.append(
          dataclasses.replace(
              shard, examples_bytes=[serialized if i == 0 else b'']
          )
      )
    yield work_units


class _MergeShards(GcsDependentDoFn):
  """Merges different shards of the same example into a single example."""

  def __init__(self, value_merger: ValueMerger, skip_faults: bool = True):
    """Initiate a _MergeShards instance.

    In case the number of input shards does not match the expected number, no
    merging will take place and the example will be thrown away.

    Args:
      value_merger: A callable that receives an Example and values sorted by
        index and returns a single merged Example.
      skip_faults: If True, faulty examples are silently dropped instead of
        causing an exception to be raised.
    """
    super().__init__()
    self._value_merger = value_merger
    self._skip_faults = skip_faults

  def process(
      self, element: list[_ExampleWorkUnitShard],
  ) -> Iterable[Example]:
    """Marges processed work units into a single Example.

    Args:
      element: A non-empty sequence of shards to be merged. It is assumed that
        all shards belong to the same example. If all_shards contains only 1
        shard, and its value is None, there is nothing to merge.

    Returns:
      In case of an only shard whose work_unit_value is None, simply returns
      the example as is.
    """

    if not element or len(element) < element[0].num_shards:
      return []

    serialized = element[0].examples_bytes[0]
    example = Example.from_bytes(serialized)  # type: ignore[arg-type]
    if len(element) == 1 and element[0].work_unit_value is None:
      return [example]

    sorted_values = [shard.work_unit_value for shard in element]
    custom_features_only_bytes = element[0].custom_features_only_bytes
    assert example is not None
    try:
      merged = self._value_merger(example, sorted_values)
      beam.metrics.Metrics.counter(self.__class__, 'merged_examples').inc()
      return [merged]
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('Caught an exception while merging shards: %s', e)
      logging.info(traceback.format_exc())
      logging.info(
          'Example merging failed. %s',
          Example.from_bytes(custom_features_only_bytes)
      )
      beam.metrics.Metrics.counter(self.__class__, 'failed_merges').inc()
      if not self._skip_faults:
        raise
    return []


def wait_for_path(path_to_wait: str, timeout: int = 900) -> None:
  start_time = time.time()
  while not os.path.exists(path_to_wait):
    if time.time() - start_time > timeout:
      raise TimeoutError(f'{path_to_wait} not ready after {timeout} seconds.')
    time.sleep(3)
