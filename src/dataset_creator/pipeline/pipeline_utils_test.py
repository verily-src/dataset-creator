"""Tests for pipeline_utils.py."""

import collections
import datetime
import functools
import os
import tempfile
import time

from absl.testing import absltest  # type: ignore[import]
import apache_beam as beam
from apache_beam.options import pipeline_options
from apache_beam.runners import direct as direct_runner
from google.cloud import pubsub_v1  # type: ignore[attr-defined]
import numpy as np

from dataset_creator import generated_dataset
from dataset_creator import test_utils
from dataset_creator.pipeline import pipeline_utils

# pylint: disable=protected-access

_PipelineOptions = pipeline_options.PipelineOptions

UNPOPULATED_TOPIC = pipeline_utils.UNPOPULATED_TOPIC
POPULATED_TOPIC = pipeline_utils.POPULATED_TOPIC
SUBSCRIPTION_ATTRIBUTE = pipeline_utils.SUBSCRIPTION_ATTRIBUTE
DATASET_ID_PUBSUB_ATTRIBUTE = pipeline_utils.DATASET_ID_PUBSUB_ATTRIBUTE
NUM_FEATURES_ATTRIBUTE = pipeline_utils.NUM_FEATURES_ATTRIBUTE


def _run_pubsub_pipeline(
    examples: list[pipeline_utils.Example],
    dataset_ids: list[str],
    num_features: int = 4,
) -> list[pipeline_utils.Example]:
  """Returns the examples the pipeline returns and the pipeline counters."""
  project = 'test-project-id'
  subscription_id = f'test_subscription_{time.time()}'

  output_messages = []
  subscriber = pubsub_v1.SubscriberClient()
  subscription_path = subscriber.subscription_path(project, subscription_id)
  subscriber.create_subscription(
      request={
          'name': subscription_path,
          'topic': subscriber.topic_path(project, POPULATED_TOPIC),
          'filter': (
              f'attributes.output_subscription = "{subscription_id}"'
          )
      }
  )

  def callback(msg):
    output_messages.append(msg)
    msg.ack()

  subscriber.subscribe(subscription_path, callback=callback)

  options = pipeline_options.PipelineOptions(flags=[], streaming=True)
  pipeline_fn = pipeline_utils.get_pubsub_streaming_pipeline(
      max_custom_features=num_features
  )
  beam.runners.DirectRunner().run_async(pipeline_fn, options)

  publisher = pubsub_v1.PublisherClient()
  for example, dataset_id in zip(examples, dataset_ids):
    attributes = {
        DATASET_ID_PUBSUB_ATTRIBUTE: dataset_id,
        SUBSCRIPTION_ATTRIBUTE: subscription_id,
        NUM_FEATURES_ATTRIBUTE: str(num_features),
    }
    publisher.publish(
        publisher.topic_path(project, UNPOPULATED_TOPIC),
        example.to_bytes(),
        **attributes
    )

  timeout = 90
  start = time.time()
  while len(output_messages) < len(examples) and start > time.time() - timeout:
    time.sleep(1)
  return output_messages


class PipelineUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    keras_model_path, outputs_layer_names, example_to_inputs = (
        test_utils.inference_parameters(224 * 224 * 3)
    )

    # Set tzinfo since inserting to spanner sets it if it's None
    now = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)
    self.generated_dataset = generated_dataset.GeneratedDataset(
        dataset_name='dataset_creator_test',
        creation_time=now,
        dataset_creator_version='test',
        get_examples_generator=functools.partial(
            test_utils.get_examples_generator,
            image_size=(224, 224),
            keras_model_path=keras_model_path,
            outputs_layer_names=outputs_layer_names,
            example_to_inputs=example_to_inputs,
        ),
    )
    self.generated_dataset.save()

  def test_batch_pipeline(self):
    with tempfile.TemporaryDirectory() as output_dir:
      output = os.path.join(output_dir, 'storage-test')
      pipeline_fn = pipeline_utils.get_batch_pipeline(
          self.generated_dataset, output
      )
      with beam.Pipeline(direct_runner.DirectRunner()) as root:
        pipeline_fn(root)

  def test_direct_streaming_pipeline(self):
    # Emulate several bundles by resetting _NUM_EXAMPLES_PER_BUNDLE:
    pipeline_utils._NUM_EXAMPLES_PER_BUNDLE = 1
    input_queue = collections.deque(
        self.generated_dataset.get_examples_generator()
    )
    output_queue = collections.deque()
    pipeline_fn = pipeline_utils.get_direct_streaming_pipeline(
        self.generated_dataset, input_queue, output_queue, num_examples=5
    )
    options = _PipelineOptions(
        flags=[],
        streaming=True,
        direct_running_mode='multi_threading',
        direct_num_workers=2,
    )
    with beam.Pipeline('BundleBasedDirectRunner', options=options) as pipeline:
      pipeline_fn(pipeline)

  def test_pubsub_streaming_pipeline(self):
    examples = list(self.generated_dataset.get_examples_generator())
    enriched_examples = []
    for i, example in enumerate(examples):
      # We enrich the Examples for coverage purposes
      enriched_examples.append(
          generated_dataset.Example(
              {'np_arr': np.array([1, 2, 3])} | dict(example) | {'index': i}
          )
      )
    output = _run_pubsub_pipeline(enriched_examples, ['test'] * len(examples))
    self.assertLen(output, len(examples))

# pylint: enable=protected-access
