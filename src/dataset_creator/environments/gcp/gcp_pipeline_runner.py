"""A module implementing PipelineRunner and MessagesHandler for GCP."""

import datetime
import json
import queue
import re
import subprocess
import threading
import time
from typing import Any, Callable, Iterable
import uuid

from apache_beam import window
import apache_beam as beam
from apache_beam.options import pipeline_options as beam_pipeline_options
from apache_beam.runners.dataflow import dataflow_runner
from google.cloud import pubsub_v1  # type: ignore[attr-defined]

from dataset_creator.environments import base_pipeline_runner
from dataset_creator.environments.gcp import gcp_utils

DEFAULT_DATAFLOW_REGION = 'us-west1'
DEFAULT_BATCH_MACHINE_TYPE = 'n2-custom-2-32768-ext'
_MAX_NUM_VCPUS = 32000
_DEFAULT_MIN_NUM_VCPUS = 64
NUM_VCPUS_PER_WORKER = 4
_MAX_NUM_WORKERS = min(int(_MAX_NUM_VCPUS / NUM_VCPUS_PER_WORKER), 4000)
_DEFAULT_MIN_NUM_WORKERS = int(_DEFAULT_MIN_NUM_VCPUS / NUM_VCPUS_PER_WORKER)

_DATAFLOW_UPDATE_MIN_NUM_WORKERS_REST_API_FORMAT = f'https://dataflow.googleapis.com/v1b3/projects/{{project_id}}/locations/{DEFAULT_DATAFLOW_REGION}/jobs/{{job_id}}?updateMask=runtime_updatable_params.min_num_workers'  # pylint: disable=line-too-long

UNPOPULATED_TOPIC = 'dataset-creator-unpopulated-queue'
POPULATED_TOPIC = 'dataset-creator-populated-queue'
SUBSCRIPTION_ATTRIBUTE = 'output_subscription'

_MAX_WAITING_MESSAGES = 20000
_MAX_WAITING_BYTES = 2 * 1024 ** 3
_TIMEOUT_BEFORE_ANY_MESSAGE = 900
_TIMEOUT_AFTER_MESSAGE = 30


def _run_command(cmd: str) -> subprocess.CompletedProcess:
  return subprocess.run(
      cmd,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=False,
  )


def get_job_id(job_name: str) -> str:
  """Returns the id of the num_features population streaming job.

  Args:
    num_features: Update the streaming job that takes care of examples with this
      many number of CustomFeatures.
  """
  gcloud_command = f'''/usr/bin/gcloud dataflow jobs list \
    --filter="STATE:Running AND NAME:{job_name}" \
    --format="value(JOB_ID)" 2>/dev/null'''
  result = _run_command(gcloud_command)
  return result.stdout.rstrip()


def schedule_returning_to_default_min_workers(job_name: str) -> None:
  """Schedula a job to periodically return min_num_workers to the default.

  The job is scheduled to run every day in 4 PM UTC time. Please note that
  requests to change the number of workers right before the job runs, might not
  take effect, as it takes a few minutes for the worker to start, and changing
  the min_num_workers value before the request was acted upon MIGHT effectively
  cancel the reqeust.

  Args:
    job_name: The job name to schedule the task for. Assumes the job is running.
  """
  scheduling_job_name = f'restores-min-workers-{job_name}'
  gcloud_command = f'''/usr/bin/gcloud scheduler jobs delete \
    {scheduling_job_name} --location={DEFAULT_DATAFLOW_REGION} --quiet \
    2>/dev/null
  '''
  _run_command(gcloud_command)

  project_number = gcp_utils.get_project_number()
  service_account = f'{project_number}-compute@developer.gserviceaccount.com'
  uri = _DATAFLOW_UPDATE_MIN_NUM_WORKERS_REST_API_FORMAT.format(
      job_id=get_job_id(job_name), project_id=gcp_utils.get_project_id(),
  )
  updatable_params = {
      'runtime_updatable_params': {'min_num_workers': _DEFAULT_MIN_NUM_WORKERS}
  }
  gcloud_command = f'''/usr/bin/gcloud scheduler jobs create http \
    {scheduling_job_name} --schedule="00 16 * * *" \
    --location="{DEFAULT_DATAFLOW_REGION}" \
    --uri={uri} --http-method=PUT \
    --message-body='{json.dumps(updatable_params)}' \
    --oauth-service-account-email={service_account} 2>/dev/null
  '''
  _run_command(gcloud_command)


def _run_dataflow_pipeline(
    job_name: str,
    pipeline: Callable[[beam.Pipeline], None],
    **pipeline_options: Any,
) -> beam.runners.runner.PipelineResult:
  project_id = gcp_utils.get_project_id()
  project_bucket = f'gs://{project_id}-dataset-creator'
  pipeline_root_dir = f'{project_bucket}/dataflow/{uuid.uuid4()}'
  region = pipeline_options.pop('region', None) or DEFAULT_DATAFLOW_REGION
  options = beam_pipeline_options.PipelineOptions(
      flags=[],
      project=project_id,
      job_name=job_name,
      region=region,
      staging_location=f'{pipeline_root_dir}/{job_name}.{time.time()}_staging',
      temp_location=f'{pipeline_root_dir}/{job_name}.{time.time()}_temp',
      pickle_library='cloudpickle',
      prebuild_sdk_container_engine='cloud_build',
      sdk_container_image=(
          f'us-west1-docker.pkg.dev/{project_id}/datasetcreator-docker/'
          'datasetcreator-image:latest'
      ),
      sdk_location='container',
      disk_size_gb=100,
      **pipeline_options,
  )
  runner = dataflow_runner.DataflowRunner()
  result = runner.run_async(transform=pipeline, options=options)
  print(
      f'Job link: https://pantheon.corp.google.com/dataflow/jobs/'
      f'{region}/{result.job_id()}?project={project_id}'
  )
  return result


class DataflowPipelineRunner(base_pipeline_runner.BasePipelineRunner):
  """An implementation of BasePipelineRunner based on Dataflow."""

  def get_batch_job_name(self, basename: str) -> str:
    """See base class."""
    now = datetime.datetime.now().strftime('%m%d-%H%M%S')
    return f'example-bank-creation-{basename.lower()}-{now}'

  def get_streaming_job_name(self, basename: str) -> str:
    """See base class."""
    project_id = gcp_utils.get_project_id()
    return f'example-dynamic-population-{project_id}-{basename}'

  def validate_job_name(self, job_name: str) -> None:
    """See base class."""
    valid_pattern = '[a-z][-a-z0-9]*[a-z0-9]'
    if not re.fullmatch(valid_pattern, job_name):
      raise ValueError(f'Invalid name! Name must conform to {valid_pattern}')

  def run_batch(
      self,
      basename: str,
      pipeline: Callable[[beam.Pipeline], None],
      **pipeline_options: Any,
  ) -> beam.runners.runner.PipelineResult:
    """See base class."""
    pipeline_options['machine_type'] = (
        pipeline_options.get('machine_type') or DEFAULT_BATCH_MACHINE_TYPE
    )
    pipeline_options['streaming'] = False
    return _run_dataflow_pipeline(
        self.get_batch_job_name(basename), pipeline, **pipeline_options,
    )

  def run_streaming(
      self,
      basename: str,
      pipeline: Callable[[beam.Pipeline], None],
      **pipeline_options: Any,
  ) -> beam.runners.runner.PipelineResult:
    """See base class."""
    pipeline_options['machine_type'] = f'n1-standard-{NUM_VCPUS_PER_WORKER}'
    pipeline_options['experiments'] = [
        f'min_num_workers={_DEFAULT_MIN_NUM_WORKERS}',
    ]
    pipeline_options['max_num_workers'] = _MAX_NUM_WORKERS
    pipeline_options['streaming'] = True

    job_name = self.get_streaming_job_name(basename)
    result = _run_dataflow_pipeline(job_name, pipeline, **pipeline_options)

    while not self.is_job_running(job_name):
      time.sleep(10)
    schedule_returning_to_default_min_workers(job_name)
    return result

  def is_job_running(self, job_name: str) -> bool:
    """See base class."""
    return bool(get_job_id(job_name))

  def update_job_min_num_workers(
      self, job_name: str, *, num_workers_to_add: int,
  ) -> int:
    """See base class."""
    job_id = get_job_id(job_name)
    gcloud_command = f'''/usr/bin/gcloud dataflow jobs describe {job_id} \
      --region="{DEFAULT_DATAFLOW_REGION}" \
      --format="value(runtimeUpdatableParams.minNumWorkers)" 2>/dev/null'''
    result = _run_command(gcloud_command)
    current_min_num_workers = int(result.stdout)

    new_min_num_workers = min(
        current_min_num_workers + num_workers_to_add, _MAX_NUM_WORKERS
    )
    new_min_num_workers = max(new_min_num_workers, _DEFAULT_MIN_NUM_WORKERS)
    gcloud_command = f'''/usr/bin/gcloud dataflow jobs update-options \
      --region="{DEFAULT_DATAFLOW_REGION}" \
      --min-num-workers={new_min_num_workers} {job_id} 2>/dev/null'''
    _run_command(gcloud_command)
    return new_min_num_workers - current_min_num_workers


def get_pubsub_subscription(
    topic_id: str, subscription_id: str, **filter_attributes: str
) -> str:
  """Returns the subscription_path of the requested subscription.

  Please note that if a subscription with this name already exists, it is
  returned regardless of whether or not the filter matches the given filter
  attributes.
  Args:
    topic_id: The topic ID to subscribe to.
    subscription_id: The name of the subscription to return.
    **filter_attributes: Mappings of attribute names and expected values. Note
      that the returned subscription filters these attributes only in case this
      helper function creates
  """
  project_id = gcp_utils.get_project_id()
  with pubsub_v1.PublisherClient() as publisher:
    topic_path = publisher.topic_path(project_id, topic_id)
    subscription_path = publisher.subscription_path(project_id, subscription_id)
    subscriptions = publisher.list_topic_subscriptions(topic=topic_path)
  if subscription_path not in subscriptions:
    with pubsub_v1.SubscriberClient() as subscriber:
      logical_attributes = [
          f'attributes.{name} = "{value}"'
          for name, value in filter_attributes.items()
      ]
      subscription_filter = ' AND '.join(logical_attributes)
      subscriber.create_subscription(
          request={
              'name': subscription_path,
              'topic': topic_path,
              'filter': subscription_filter
          }
      )
  return subscription_path


class PubsubMessagesHandler(base_pipeline_runner.BasePipelineMessagesHandler):
  """An implementation of BasePipelineMessagesHandler based on Pub/Sub."""

  def __init__(self):
    super().__init__()
    self._num_written_to_pubsub = 0
    self._num_read_from_pubsub = 0
    self._keep_writing_to_pubsub = True
    self._subscription_id = f'dynamic-dataset-{time.time()}'

  def send_messages_to_pipeline(
      self, messages: Iterable[bytes], **attributes: str
  ) -> None:
    """See base class."""
    # Add the subscription id to the attributes so we can pull from this
    # specific id later on.
    attributes[SUBSCRIPTION_ATTRIBUTE] = self._subscription_id
    condition = threading.Condition()

    def wait_predicate() -> bool:
      num_waiting = self._num_written_to_pubsub - self._num_read_from_pubsub
      return num_waiting < _MAX_WAITING_MESSAGES

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(
        gcp_utils.get_project_id(), UNPOPULATED_TOPIC
    )

    self._keep_writing_to_pubsub = True
    for message in messages:
      if not self._keep_writing_to_pubsub:
        break
      with condition:
        condition.wait_for(wait_predicate, timeout=1)
      publisher.publish(topic_path, data=message, **attributes)
      self._num_written_to_pubsub += 1
    self._keep_writing_to_pubsub = False

  def recv_messages_from_pipeline(
      self, **filter_attributes: str
  ) -> Iterable[bytes]:
    """See base class."""
    # Since we start recving before starting to send, this is the right place
    # to reset the counters.
    self._num_written_to_pubsub = 0
    self._num_read_from_pubsub = 0

    subscription_path = get_pubsub_subscription(
        POPULATED_TOPIC, self._subscription_id, **filter_attributes
    )

    output_queue: queue.Queue[bytes] = queue.Queue()

    def pull_message_from_pubsub(message):
      attribute = message.attributes[SUBSCRIPTION_ATTRIBUTE]
      if attribute == self._subscription_id:
        output_queue.put(message.data)
        self._num_read_from_pubsub += 1
      message.ack()

    with pubsub_v1.SubscriberClient() as subscriber:
      subscriber.subscribe(
          subscription_path,
          flow_control=pubsub_v1.types.FlowControl(
              max_bytes=_MAX_WAITING_BYTES, max_messages=_MAX_WAITING_MESSAGES,
          ),
          callback=pull_message_from_pubsub,
      )

      timeout = _TIMEOUT_BEFORE_ANY_MESSAGE
      while True:
        try:
          yield output_queue.get(block=True, timeout=timeout)
          timeout = _TIMEOUT_AFTER_MESSAGE
        except (queue.Empty, KeyboardInterrupt) as e:
          if (
              isinstance(e, KeyboardInterrupt) or
              (not self._keep_writing_to_pubsub)
          ):
            break

  def create_read_messages_ptransform(
      self, **filter_attributes: str
  ) -> beam.PTransform:
    """See base class."""
    class _PubSubExampleReader(beam.PTransform):
      """Reads from Pub/Sub Examples with a given number of CustomFeatures."""

      def expand(self, input_or_inputs: beam.PCollection) -> beam.PCollection:
        filter_id = '-'.join(filter_attributes.values())
        subscription_id = f'subscription-with-{filter_id}-params'

        # Create a subscription that only contains Examples with `num_features`
        # CustomFeatures.
        subscription_path = get_pubsub_subscription(
            UNPOPULATED_TOPIC, subscription_id, **filter_attributes
        )
        return (
            input_or_inputs
            | 'PubSubRead' >> beam.io.ReadFromPubSub(
                subscription=subscription_path, with_attributes=True
            )
            | 'OutputMessages' >> beam.Map(
                lambda message: (message.data, message.attributes)
            )
            | 'WindowInto' >> beam.WindowInto(window.FixedWindows(.1))
        )
    return _PubSubExampleReader()

  def create_write_out_messages_ptransform(
      self, **attrtibutes
  ) -> beam.PTransform:
    """See base class."""
    class _PubSubSubscriptionWriter(beam.PTransform):
      """"Writes examples back to Pub/Sub."""

      def expand(self, input_or_inputs: beam.PCollection) -> beam.PCollection:
        return (
            input_or_inputs
            | 'ToPubsubMessages' >> beam.Map(
                lambda m: beam.io.PubsubMessage(data=m, attributes=attrtibutes)
            )
            | 'WriteToPubSub' >> beam.io.WriteToPubSub(
                pubsub_v1.PublisherClient.topic_path(
                    gcp_utils.get_project_id(), POPULATED_TOPIC
                ),
                with_attributes=True
            )
        )
    return _PubSubSubscriptionWriter()
