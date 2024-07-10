"""Tests for beam_populator.py."""

import pathlib
import threading
import time
from typing import Any, Callable, Optional, Sequence, Union
import uuid

from absl.testing import parameterized  # type: ignore[import]
import apache_beam as beam
from apache_beam.options import pipeline_options
from apache_beam.runners import direct as direct_runner
from apache_beam.runners.interactive import interactive_beam
from apache_beam.runners.interactive import interactive_runner
import numpy as np

from dataset_creator import example_lib
from dataset_creator import test_utils
from dataset_creator.features import images_feature
from dataset_creator.pipeline import beam_populator

# pylint: disable=protected-access

Example = example_lib.Example

_IMAGE_KEY = 'image'
_MERGED_KEY = 'test'


def _default_merge_values(example: Example, values: Sequence[Any]) -> Example:
  """Merges multiple values to an entry _MERGED_KEY of a new example.

  Args:
    example: The example to merge into.
    values: The values to be merged into the new example.

  Returns:
    A new example with values under the _MERGED_KEY key in the new example.
  """
  return Example(example | {_MERGED_KEY: values})


def _get_beam_populator(
    example_classifier: Optional[beam_populator.DatasetClassifier] = None,
    operation_id_assigner: Optional[beam_populator.OperationIdAssigner] = None,
    splitter: Optional[beam_populator.ExampleSplitter] = None,
    context_setup: Optional[beam_populator.ContextSetup] = None,
    value_processor: Optional[beam_populator.ValueProcessorWithContext] = None,
    value_merger: Optional[beam_populator.ValueMerger] = None,
    skip_faults: bool = True,
) -> beam_populator.ExamplePopulator:
  """Returns a populator that populates the given examples.

  Args:
    splitter: Splits an example to multiple values, each is an individual
      processing input. If the sequence is empty, no processing takes place. If
      None, using a default splitter of (lambda _: range(3)).
    context_setup: Creates a context for processing. If None, using a default of
      (lambda _: None) for context.
    value_processor: Processes a value using some common context, to create a
      new value. A return value of None indicates a failure in processing. If
      None, using a default of (lambda value, _: value).
  """
  id_assigner = operation_id_assigner or (lambda _: [str(uuid.uuid4())])
  return beam_populator.ExamplePopulator(
      dataset_classifier=example_classifier or (lambda _: ''),
      operation_id_assigner=id_assigner,
      splitter=splitter or (lambda _: range(3)),
      context_setup=context_setup or (lambda _: None),
      value_processor=value_processor or (lambda _, value, _2: value),
      value_merger=value_merger or _default_merge_values,
      skip_faulty_examples=skip_faults
  )

def _get_population_pipeline(
    examples: Union[Sequence[Example], beam.PTransform],
    populator: beam_populator.ExamplePopulator
) -> Callable[[beam.Pipeline], beam.PCollection[Example]]:
  if isinstance(examples, Sequence):
    examples = beam.Create(examples)
  return lambda root: (root | examples | 'test' >> populator)


def _run_population_pipeline(
    runner: beam.runners.PipelineRunner,
    examples: Union[Sequence[Example], beam.PTransform],
    options: Optional[pipeline_options.PipelineOptions] = None,
    **kwargs
) -> tuple[beam.runners.runner.PipelineResult, beam.PCollection]:
  populator = _get_beam_populator(**kwargs)
  population_pipeline = _get_population_pipeline(examples, populator)
  pipeline = beam.Pipeline(runner, options=options)
  output_pcollection = population_pipeline(pipeline)
  if isinstance(runner, interactive_runner.InteractiveRunner):
    interactive_beam.watch(locals())
  result = pipeline.run()
  result.wait_until_finish()
  return result, output_pcollection


def _get_population_counters(
    examples: Union[Sequence[Example], beam.PTransform],
    options: Optional[pipeline_options.PipelineOptions] = None,
    **kwargs
) -> dict[str, int]:
  """Returns the counters for the population process."""
  result, _ = _run_population_pipeline(
      direct_runner.DirectRunner(), examples, options=options, **kwargs
  )
  counters = result.metrics().query()['counters']
  return {counter.key.metric.name: counter.committed for counter in counters}


def _get_population_output(
    examples: Union[Sequence[Example], beam.PTransform],
    options: Optional[pipeline_options.PipelineOptions] = None,
    **kwargs,
) -> list[Example]:
  """Returns a list of the output examples of the population process."""
  result, output_pcollection = _run_population_pipeline(
      interactive_runner.InteractiveRunner(),
      examples,
      options=options,
      **kwargs
  )
  return result.get(output_pcollection)


def _extract_timestamp_millis(example: Example) -> int:
  return example[_IMAGE_KEY]._read_at[0]  # type: ignore[union-attr]


def _get_example(timestamp_millis: int = 0) -> Example:
  video_path = test_utils.mock_video_path()
  return Example({
      _IMAGE_KEY: images_feature.ImagesFeature(
          reader=images_feature.get_default_reader(video_path),
          read_by=images_feature.READ_BY_TIMESTAMP_MILLIS,
          read_at=[timestamp_millis],
      )
  })

class BeamPopulatorTest(parameterized.TestCase):

  def setUp(self):
    beam_populator._NUM_CONTEXT_SETUPS_FOR_TESTING = 0

  def test_total_counter(self):
    counters = _get_population_counters(
        examples=[_get_example()], splitter=lambda _: range(3)
    )
    self.assertEqual(counters['processing_work_units'], 3)

  def test_populator_drops_example_with_empty_values_to_process(self):
    output_examples = _get_population_output(
        examples=[_get_example()], splitter=lambda _: [],
    )
    self.assertEmpty(output_examples)

  def test_population_with_single_input_value(self):
    output_examples = _get_population_output(
        examples=[_get_example()], splitter=lambda _: [5],
    )
    self.assertEqual(output_examples[0][_MERGED_KEY], [5])

  def test_success_and_fail_counters(self):
    counters = _get_population_counters(
        examples=[_get_example()],
        splitter=lambda _: [0, 1, 2],
        # processed value of None is failure - fail on odd values:
        value_processor=lambda _, value, _2: None if value % 2 else value,
    )
    self.assertEqual(counters['completed_work_units'], 2)
    self.assertEqual(counters['failed_work_units'], 1)

  def test_population(self):
    output_examples = _get_population_output(
        examples=[_get_example(), _get_example(1)],
        splitter=lambda _: range(3),
        # Population process that will add 1 to inputs + merge the default way
        context_setup=lambda _: 1,
        value_processor=lambda _, value, context: value + context,
    )
    self.assertLen(output_examples, 2)
    self.assertEqual(output_examples[0][_MERGED_KEY], [1, 2, 3])

  def test_population_drops_the_example_when_some_inputs_fail(self):
    output_examples = _get_population_output(
        examples=[_get_example()],
        splitter=lambda _: [0, 1, 2],
        value_processor=lambda _, value, _2: None if value % 2 else value,
    )
    self.assertEmpty(output_examples)

  def test_pipeline_succeeds_with_mutating_fn_inputs(self):
    def mutate_input(example, value, _):
      del example
      # Explicitly mutate the input value:
      value[0] = 4
      return int(value[0])

    _get_population_counters(
        examples=[_get_example()],
        splitter=lambda _: [np.array([1, 2, 3])],
        value_processor=mutate_input,
    )

  @parameterized.named_parameters(
      ('on_example_classification', {'example_classifier': (lambda _: 1/0)}),
      ('on_split', {'splitter': (lambda _: 1/0)}),
      ('on_process', {'value_processor': (lambda _, _2, _3: 1/0)}),
      ('on_merge', {'value_merger': (lambda _, _2: 1/0)}),
      ('on_process_of_some_shards', {
          'splitter': lambda _: [0, 1, 2],
          'value_processor': (lambda _, value, _2: 1/0 if value % 2 else 0)
      }),
  )
  def test_pipeline_skips_examples_with_errors(self, pipeline_kwargs):
    self.assertEmpty(
        _get_population_output(
            examples=[_get_example()],
            skip_faults=True,
            **pipeline_kwargs
        )
    )

  @parameterized.named_parameters(
      ('on_example_classification', {'example_classifier': (lambda _: 1/0)}),
      ('on_split', {'splitter': (lambda _: 1/0)}),
      ('on_process', {'value_processor': (lambda _, _2, _3: 1/0)}),
      ('on_merge', {'value_merger': (lambda _, _2: 1/0)}),
  )
  def test_pipeline_raises_on_errors_when_not_skipping(self, pipeline_kwargs):
    with self.assertRaises(ZeroDivisionError):
      _get_population_output(
          examples=[_get_example()],
          skip_faults=False,
          **pipeline_kwargs
      )

  def test_pipeline_with_examples_from_several_datasets(self):
    examples = [
        _get_example(timestamp_millis=0),
        _get_example(timestamp_millis=1)
    ]

    def classifier(example):
      return str(_extract_timestamp_millis(example))

    output = _get_population_output(
        examples=examples,
        example_classifier=classifier,
        context_setup=_extract_timestamp_millis,
        value_processor=lambda _, value, context: value + context
    )
    self.assertLen(output, 2)
    [output_0] = [r for r in output if classifier(r) == '0']
    [output_1] = [r for r in output if classifier(r) == '1']
    self.assertEqual(output_0[_MERGED_KEY], [0, 1, 2])
    self.assertEqual(output_1[_MERGED_KEY], [1, 2, 3])

  def test_example_fails_if_a_single_work_unit_processing_fails(self):
    example = _get_example()
    output = _get_population_output(
        [example],
        splitter=lambda _: range(3),
        value_processor=lambda _, value, _2: 1 / value  # Only 0 fails
    )
    self.assertEmpty(output)

  def test_wait_for_path_raises_after_timeout(self):
    with self.assertRaisesRegex(TimeoutError, 'not ready after 2 seconds.'):
      beam_populator.wait_for_path('this/path/doesnt/exist', 2)

  def test_wait_for_path_doesnt_raise_when_path_created_before_timeout(self):
    path_to_touch = 'file.txt'

    def delayed_touch(delay: int = 3):
      time.sleep(delay)
      pathlib.Path(path_to_touch).touch()

    thread = threading.Thread(target=delayed_touch)
    thread.start()
    beam_populator.wait_for_path(path_to_touch, 60)
    thread.join()

  def test_same_op_id_examples_are_only_populated_once(self):
    examples = [_get_example() for _ in range(5)]
    counters = _get_population_counters(
        examples,
        operation_id_assigner=lambda _: [0],
        splitter=lambda _: range(3),
        value_processor=lambda _, value, _2: value
    )
    # Assert that only 1 example (that was split in 3) was processed
    self.assertEqual(counters['processing_work_units'], 3)
    # Assert that all original 5 examples were merged
    self.assertEqual(counters['merged_examples'], 5)

# pylint: enable=protected-access
