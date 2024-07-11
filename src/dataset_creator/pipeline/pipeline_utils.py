"""A module which exposes all the required pipeline operations.

This module builds a pipeline from the pipeline parameters (see documentation
of pipeline_utils.get_batch_pipeline). The pipeline pulls all examples from
the DatasetSequenceExamples spanner tables, samples the requested timestamps,
converts them to tf.train.Example-s, and saves them into a table to serve as an
example bank.
"""

import collections
import functools
import math
import threading
import time
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from apache_beam import window
import apache_beam as beam
from apache_beam.io.gcp.experimental import spannerio
from apache_beam.testing import test_stream
from google.cloud import pubsub_v1  # type: ignore[attr-defined]
from google.cloud import spanner  # type: ignore[attr-defined]
import more_itertools
import numpy as np
from typing_extensions import TypeAlias

from dataset_creator import example_lib
from dataset_creator import generated_dataset
from dataset_creator import helpers
from dataset_creator.features import base_feature
from dataset_creator.pipeline import beam_populator
from dataset_creator.pipeline import example_bank_sink

Example = example_lib.Example
PCollection: TypeAlias = beam.pvalue.PCollection

# ================= GLOBALS FOR THE DIRECT STREAMING PIPELINE =================

_NUM_EXAMPLES_PER_BUNDLE = 512
_MAX_EXAMPLES_IN_STREAM = 5000000  # 5M
_MAX_WAIT_BETWEEN_EXAMPLES = 10
_MAX_WAITING_OUTPUT_EXAMPLES = 10000

_INPUT_QUEUE: Optional[collections.deque[Example]] = None
_OUTPUT_QUEUE: Optional[collections.deque[Example]] = None

# =============== CONSTANTS FOR THE DATAFLOW STREAMING PIPELINE ================

_DATASET_ID_KEY = '_DATASET_ID_KEY_FOR_STREAMING'
_OUTPUT_SUBSCRIPTION_KEY = '_OUTPUT_SUBSCRIPTION_FOR_STREAMING'

DATASET_ID_PUBSUB_ATTRIBUTE = 'dataset_id'
SUBSCRIPTION_ATTRIBUTE = 'output_subscription'
NUM_FEATURES_ATTRIBUTE = 'num_custom_features'
UNPOPULATED_TOPIC = 'dataset-creator-unpopulated-queue'
POPULATED_TOPIC = 'dataset-creator-populated-queue'


class _ReadFromSpanner(beam.PTransform):
  """Reads from spanner by dividing all Examples into shards, then flattens."""

  def __init__(
      self,
      saved_dataset: generated_dataset.GeneratedDataset,
  ):
    super().__init__()
    self._saved_dataset = saved_dataset

  def expand(self, input_or_inputs: PCollection) -> PCollection:
    """Reads all the examples belonging to saved_dataset into a PCollection.

    Args:
      input_or_inputs: The pipeline to use for reading.

    Returns:
      A PCollection of all examples belonging to saved_dataset.
    """
    *_, instance_id, _, database_id = helpers.get_db().name.split('/')
    key_range = spanner.KeyRange(
        start_closed=[*self._saved_dataset.key, 0],
        end_open=[*self._saved_dataset.key, self._saved_dataset.num_examples],
    )
    return (
        input_or_inputs
        | 'ReadFromSpanner' >> spannerio.ReadFromSpanner(
            project_id=helpers.get_project_id(),
            instance_id=instance_id,
            database_id=database_id,
            table=generated_dataset.EXAMPLES_TABLE,
            columns=['EncodedExample'],
            keyset=spannerio.KeySet(ranges=[key_range]),
        )
        | 'Reshuffle' >> beam.Reshuffle()
        | 'DecodeExamples' >> beam.Map(lambda x: Example.from_db_encoded(x[0]))
    )


class _DirectStreamingGeneratorExampleReader(beam.PTransform):
  """Reads Examples from the global _INPUT_QUEUE."""

  def __init__(self, num_examples: int):
    super().__init__()
    self._num_examples = num_examples

  def expand(self, input_or_inputs: PCollection) -> PCollection:
    # This method's implementation uses TestStream, as it is the only current
    # way for streaming with the DirectRunner.
    # TODO(itayr): Change the implementation to using BoundedSources /
    #              SplittableDoFns once the following issue is resolved:
    #              https://github.com/apache/beam/issues/26577
    indices_stream = test_stream.TestStream().with_output_types(int)
    num_bundles = int(math.ceil(self._num_examples / _NUM_EXAMPLES_PER_BUNDLE))
    for _ in range(num_bundles):
      indices_stream.add_elements([_NUM_EXAMPLES_PER_BUNDLE])

    def read_from_queue(max_elements_to_read: int) -> Iterable[Example]:
      total = 0
      last_input_queue_not_empty = time.time()
      while (
          time.time() - last_input_queue_not_empty < _MAX_WAIT_BETWEEN_EXAMPLES
      ):
        if not _INPUT_QUEUE:
          time.sleep(1)  # Give the CPU a chance to context switch
          continue
        last_input_queue_not_empty = time.time()
        yield _INPUT_QUEUE.popleft()
        total += 1
        if total == max_elements_to_read:
          break

    return (
        input_or_inputs
        | 'CreateIndicesStream' >> indices_stream
        | 'WindowInto' >> beam.WindowInto(window.FixedWindows(.1))
        | 'ReadFromQueue' >> beam.FlatMap(read_from_queue)
    )


class _PubSubExampleReader(beam.PTransform):
  """Reads from Pub/Sub Examples with a given number of CustomFeatures."""

  def __init__(self, num_features: int):
    super().__init__()
    self._num_features = num_features

  def expand(self, input_or_inputs: PCollection) -> PCollection:
    subscription_id = f'examples-with-{self._num_features}-custom-features'

    # Create a subscription that only contains Examples with `num_features`
    # CustomFeatures.
    filter_attributes = {NUM_FEATURES_ATTRIBUTE: str(self._num_features)}
    subscription_path = helpers.get_pubsub_subscription(
        UNPOPULATED_TOPIC, subscription_id, **filter_attributes
    )
    return (
        input_or_inputs
        | 'PubSubRead' >> beam.io.ReadFromPubSub(
            subscription=subscription_path, with_attributes=True
        )
        | 'NormalizeExamples' >> beam.Map(_extract_example_from_pubsub_message)
        | 'WindowInto' >> beam.WindowInto(window.FixedWindows(.1))
    )


def _extract_example_from_pubsub_message(
    message: beam.io.PubsubMessage
) -> Example:
  example = Example.from_bytes(message.data)
  attributes = message.attributes
  added_attributes = {
      _DATASET_ID_KEY: attributes[DATASET_ID_PUBSUB_ATTRIBUTE],
      _OUTPUT_SUBSCRIPTION_KEY: attributes[SUBSCRIPTION_ATTRIBUTE],
  }
  return Example(example | added_attributes)


class _HeterogeneousExampleProcessor(beam.PTransform):
  """Populates Examples from different datasets with all features.

  Unlike _HomogenousExampleProcessor, this PTransform supports a PCollection
  with examples from multiple datasets, and Examples that don't conform to the
  same template, in particular.
  This PTransform assumes Examples in the incoming PCollection have an entry
  _DATASET_ID_KEY which gives an id of the dataset this Example belongs to.
  Since Examples don't share a joint template, we build the PTransform to
  populate CustomFeature based on the CustomFeature's index, up to a given
  maximum number of CustomFeatures per example.
  """

  def __init__(
      self,
      *,
      skip_faults: bool = True,
      max_custom_features: int = 4,
      index_to_label: Optional[Callable[[int], str]] = None,
      example_classifier: Optional[beam_populator.DatasetClassifier] = None,
  ):
    """Instantiates a HeterogeneousExampleProcessor.

    Args:
      skip_faults: Whether the pipeline should skip faults or raise on fault.
      max_custom_features: The maximal number of CustomFeatures to populate in
        each example of the PCollection.
      index_to_label: An optional callable which converts the CustomFeature's
        index to the population stage label. Default is str.
      example_classifier: A beam_populator.ExampleClassifier which is used to
        extract the dataset_id for each example.
    """
    super().__init__()
    self._skip_faults = skip_faults
    self._max_custom_features = max_custom_features
    self._index_to_label = index_to_label or str
    self._example_classifier: beam_populator.DatasetClassifier = (
        example_classifier or (lambda example: example[_DATASET_ID_KEY])  # type: ignore[assignment,return-value]  # pylint: disable=line-too-long
    )

  def expand(self, input_or_inputs: beam.PCollection) -> beam.PCollection:
    """Processes all the given examples using all relevant ExamplePopulators.

    Args:
      input_or_inputs: The PCollection to be processed.

    Returns:
      An Example PCollection with all examples containing all processed values.
    """
    processed_examples = input_or_inputs
    for i in range(self._max_custom_features):
      populator = beam_populator.ExamplePopulator(
          dataset_classifier=self._example_classifier,
          operation_id_assigner=functools.partial(
              _assign_operation_id_by_index, index=i
          ),
          splitter=functools.partial(_split_by_index, index=i),
          context_setup=functools.partial(_context_setup_by_index, index=i),
          value_processor=functools.partial(_process_by_index, index=i),
          value_merger=functools.partial(_merge_by_index, index=i),
          skip_faulty_examples=self._skip_faults,
      )
      processed_examples = (
          processed_examples | self._index_to_label(i) >> populator
      )

    return processed_examples | 'Finalize' >> beam.Map(
        lambda example: example.finalize()
    )


def _get_custom_feature_by_index(
    example: Example, index: int
) -> tuple[str, Optional[base_feature.CustomFeature]]:
  """Returns the (user_key, feature) tuple of the i-th CustomFeature.

  In case the example doesn't have index CustomFeatures, ('', None) is returned.

  Args:
      example: The example to extract CustomFeatures from.
      index: The index of the CustomFeature among all CustomFeatures in example.
  """
  i = 0
  for user_key, feature in example.items():
    if isinstance(feature, base_feature.CustomFeature):
      if i == index:
        return user_key, feature
      i += 1
  return '', None


def _assign_operation_id_by_index(
    example: Example, index: int
) -> tuple[base_feature.ValueFeature, ...]:
  feature_key, feature = _get_custom_feature_by_index(example, index)
  if not feature:
    return ()
  # GenericLambdaFeatures depend on previous features for their population, so
  # in that case we need to pass a list of all previous hashes. Otherwise, the
  # CustomFeature depends only on itself, so it suffices to pass its hash.
  if feature.is_self_contained:
    return (hash(feature),)
  depends_on = []
  for other_key, other_feature in example.items():
    if isinstance(other_feature, base_feature.CustomFeature):
      other_feature = hash(other_feature)
    elif isinstance(other_feature, np.ndarray):
      other_feature = other_feature.tolist()
    depends_on.append(other_feature)
    if other_key == feature_key:
      break
  return tuple(depends_on)


def _split_by_index(example: Example, index: int) -> Any:
  _, feature = _get_custom_feature_by_index(example, index)
  return [None] if feature is None else feature.split()


def _context_setup_by_index(example: Example, index: int) -> Any:
  _, feature = _get_custom_feature_by_index(example, index)
  return None if feature is None else feature.create_context()


def _process_by_index(
    example: Example, metadata_value: Any, context: Any, index: int
) -> Any:
  _, feature = _get_custom_feature_by_index(example, index)
  assert feature is not None
  return feature.process(metadata_value, context)


def _merge_by_index(
    example: Example, values: Sequence[Any], index: int
) -> Example:
  user_key, feature = _get_custom_feature_by_index(example, index)
  assert feature is not None
  merged = example[user_key].merge(values)  # type: ignore[union-attr]
  merged_outputs = {
      example_lib.nested_key(user_key, output_key): output
      for output_key, output in merged.items()
  }
  return Example(example | merged_outputs)


class _HomogenousExampleProcessor(_HeterogeneousExampleProcessor):
  """Populates Examples from the same dataset with all features.

  This PTransform assumes all Examples in the incoming PCollection belong to the
  same dataset.
  """

  def __init__(
      self,
      saved_dataset: generated_dataset.GeneratedDataset,
      skip_faults: bool = True,
  ):
    assert saved_dataset.get_examples_generator is not None
    template = more_itertools.first(saved_dataset.get_examples_generator())
    num_custom_features = 0
    for feature in template.values():
      if isinstance(feature, base_feature.CustomFeature):
        num_custom_features += 1
    super().__init__(
        skip_faults=skip_faults,
        max_custom_features=num_custom_features,
        index_to_label=(lambda i: _get_custom_feature_by_index(template, i)[0]),
        example_classifier=(lambda _: ''),
    )


class _RecordExampleWriter(beam.PTransform):
  """Writes examples to storage."""

  def __init__(self, output_path: str):
    super().__init__()
    self._output_path = output_path

  def expand(self, input_or_inputs: PCollection) -> None:
    """Saves the given examples to storage.

    Args:
      input_or_inputs: The examples to be saved.
    """
    _ = (
        input_or_inputs
        | 'WriteToStorage' >> beam.io.Write(
            example_bank_sink.ShardFileSink(self._output_path)
        )
    )


class _DirectStreamingExampleWriter(beam.DoFn):
  def process(self, element: Example) -> None:
    condition = threading.Condition()
    with condition:
      condition.wait_for(
          lambda: len(_OUTPUT_QUEUE) < _MAX_WAITING_OUTPUT_EXAMPLES,  # type: ignore[arg-type]  # pylint: disable=line-too-long
          timeout=1,
      )
    assert _OUTPUT_QUEUE is not None
    _OUTPUT_QUEUE.append(element)


class _PubSubSubscriptionWriter(beam.PTransform):
  """"Writes examples back to Pub/Sub, adding the attribute for subscribers."""

  def expand(self, input_or_inputs: PCollection) -> PCollection:
    def to_pubsub_message(example: Example) -> beam.io.PubsubMessage:
      output_subscription = example[_OUTPUT_SUBSCRIPTION_KEY]
      example_to_send = Example(
          {
              k: v for k, v in example.items()
              if k not in [_DATASET_ID_KEY, _OUTPUT_SUBSCRIPTION_KEY]
          }
      )
      return beam.io.PubsubMessage(
          data=example_to_send.to_bytes(),
          attributes={SUBSCRIPTION_ATTRIBUTE: output_subscription}
      )

    return (
        input_or_inputs
        | 'ToPubsubMessages' >> beam.Map(to_pubsub_message)
        | 'WriteToPubSub' >> beam.io.WriteToPubSub(
            pubsub_v1.PublisherClient.topic_path(
                helpers.get_project_id(), POPULATED_TOPIC
            ),
            with_attributes=True
        )
    )


def get_direct_streaming_pipeline(
    saved_dataset: generated_dataset.GeneratedDataset,
    input_queue: collections.deque[Example],
    output_queue: collections.deque[Example],
    num_examples: Optional[int] = None,
    skip_faulty_examples: bool = True,
) -> Callable[[Union[beam.Pipeline, beam.PCollection]], None]:
  """Returns the direct streaming pipeline to be run with ApacheBeam.

  Args:
    saved_dataset: The dataset whose example bank is created in the pipeline.
    input_queue: The queue to pull unpopulated examples from.
    output_queue: The queue to push populated examples to.
    num_examples: The number of examples to populate in the pipeline.
    skip_faulty_examples: If False, an example causing an exception will raise
        the exception, instead of silently dropped. Default is True. PLEASE NOTE
        that setting this value to False may cause the entire pipeline to fail!
  """
  def direct_streaming_pipeline(root: Union[beam.Pipeline, beam.PCollection]):
    # pylint: disable-next=global-variable-not-assigned
    global _INPUT_QUEUE, _OUTPUT_QUEUE, _MAX_EXAMPLES_IN_STREAM
    _INPUT_QUEUE = input_queue
    _OUTPUT_QUEUE = output_queue

    _ = (
        root
        | 'Read' >> _DirectStreamingGeneratorExampleReader(
            num_examples or _MAX_EXAMPLES_IN_STREAM
        )
        | 'Populate' >> _HomogenousExampleProcessor(
            saved_dataset, skip_faulty_examples
        )
        | 'Write' >> beam.ParDo(_DirectStreamingExampleWriter())
    )
  return direct_streaming_pipeline


def get_pubsub_streaming_pipeline(
    skip_faulty_examples: bool = True,
    max_custom_features: int = 8,
) -> Callable[[Union[beam.Pipeline, beam.PCollection]], None]:
  """Returns the dataflow streaming pipeline to be run with ApacheBeam.

  This pipeline expects reading Examples from Pub/Sub topic given by the global
  UNPOPULATED_TOPIC. Each PubsubMessage must have the following 2 attributes:
    1. DATASET_ID_PUBSUB_ATTRIBUTE: A dataset_id this Example belongs to.
    2. OUTPUT_SUBSCRIPTION_ATTRIBUTE: An attribute that will be kept as the
      incoming Example for the populated example. This allows publishers to
      subscribe to their own Examples only.
  Populated Examples are written to the POPULATED_TOPIC of Pub/Sub.

  Args:
    skip_faulty_examples: If False, an example causing an exception will raise
        the exception, instead of silently dropped. Default is True. PLEASE NOTE
        that setting this value to False may cause the entire pipeline to fail!
    max_custom_features: The number of population steps the pipeline allows.
  """
  def dataflow_streaming_pipeline(root: Union[beam.Pipeline, beam.PCollection]):
    _ = (
        root
        | 'Read' >> _PubSubExampleReader(num_features=max_custom_features)
        | 'Populate' >> _HeterogeneousExampleProcessor(
            skip_faults=skip_faulty_examples,
            max_custom_features=max_custom_features,
        )
        | 'Write' >> _PubSubSubscriptionWriter()
    )
  return dataflow_streaming_pipeline


def get_batch_pipeline(
    saved_dataset: generated_dataset.GeneratedDataset,
    output_path: str,
    skip_faulty_examples: bool = True,
) -> Callable[[Union[beam.Pipeline, beam.PCollection]], None]:
  """Returns the sampling pipeline to be run with ApacheBeam.

  Args:
    saved_dataset: The dataset whose example bank is created in the pipeline.
    output_path: The path to save the resulting examples to.
    skip_faulty_examples: If False, an example causing an exception will raise
        the exception, instead of silently dropped. Default is True. PLEASE NOTE
        that setting this value to False may cause the entire pipeline to fail!
  """

  def batch_pipeline(root: Union[beam.Pipeline, beam.PCollection]) -> None:
    _ = (
        root
        | 'Read' >> _ReadFromSpanner(saved_dataset)
        | 'Populate' >> _HomogenousExampleProcessor(
            saved_dataset, skip_faulty_examples
        )
        | 'Write' >> _RecordExampleWriter(output_path)
    )
  return batch_pipeline
