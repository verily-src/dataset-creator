"""A binary which wraps the beam pipeline, and so allows running it on borg.

The binary receives command line flags to configure the pipeline. If necessary,
it creates the directories needed by the output path.

The dataflow python client (google-cloud-dataflow-client) is still very iffy, so
unfortunately we have to perform some actions by invoking the gcloud command.
"""

import datetime
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Callable, Sequence

import apache_beam as beam
from apache_beam.options import pipeline_options
from apache_beam.runners.dataflow import dataflow_runner

from dataset_creator import generated_dataset
from dataset_creator import helpers
from dataset_creator.pipeline import pipeline_utils

DEFAULT_DATAFLOW_REGION = 'us-west1'
DEFAULT_BATCH_MACHINE_TYPE = 'n2-custom-2-32768-ext'
_MAX_NUM_VCPUS = 32000
_DEFAULT_MIN_NUM_VCPUS = 64
NUM_VCPUS_PER_WORKER = 4
_MAX_NUM_WORKERS = min(int(_MAX_NUM_VCPUS / NUM_VCPUS_PER_WORKER), 4000)
_DEFAULT_MIN_NUM_WORKERS = int(_DEFAULT_MIN_NUM_VCPUS / NUM_VCPUS_PER_WORKER)

_DATAFLOW_UPDATE_MIN_NUM_WORKERS_REST_API_FORMAT = f'https://dataflow.googleapis.com/v1b3/projects/{{project_id}}/locations/{DEFAULT_DATAFLOW_REGION}/jobs/{{job_id}}?updateMask=runtime_updatable_params.min_num_workers'  # pylint: disable=line-too-long


def _run_command(cmd: str) -> subprocess.CompletedProcess:
  return subprocess.run(
      cmd,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=False,
  )


def get_batch_job_name(dataset_name: str) -> str:
  now = datetime.datetime.now().strftime('%m%d-%H%M%S')
  return f'example-bank-creation-{dataset_name.lower()}-{now}'


def get_streaming_job_name(num_features: int) -> str:
  project_id = helpers.get_project_id()
  return f'example-dynamic-population-{project_id}-{num_features}'


def validate_job_name(job_name: str) -> None:
  valid_job_pattern = '[a-z][-a-z0-9]*[a-z0-9]'
  if not re.fullmatch(valid_job_pattern, job_name):
    raise ValueError(f'Invalid name! Name must conform to {valid_job_pattern}')


def _run_dataflow_pipeline(
    pipeline: Callable[[beam.Pipeline], None],
    job_name: str,
    pipeline_workdir: str,
    region: str = DEFAULT_DATAFLOW_REGION,
    **additional_pipeline_options,
) -> beam.runners.runner.PipelineResult:
  project_id = helpers.get_project_id()
  project_bucket = f'gs://{project_id}-dataset-creator'
  pipeline_root_dir = f'{project_bucket}/{pipeline_workdir}'
  options = pipeline_options.PipelineOptions(
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
      **additional_pipeline_options,
  )
  runner = dataflow_runner.DataflowRunner()
  result = runner.run_async(transform=pipeline, options=options)
  print(
      f'Job link: https://pantheon.corp.google.com/dataflow/jobs/'
      f'{region}/{result.job_id()}?project={project_id}'
  )
  return result


def run_batch(
    dataset_name: str,
    creation_time: datetime.datetime,
    output_path: str,
    additional_pip_requirements: Sequence[str] = (),
    region: str = DEFAULT_DATAFLOW_REGION,
    machine_type: str = DEFAULT_BATCH_MACHINE_TYPE,
) -> beam.runners.runner.PipelineResult:
  """Runs a batch population pipeline for the dataset given by the arguments.

  Args:
    dataset_name: The name of the dataset to populate.
    creation_time: The creation_time of the dataset to populate.
    output_path: The path to write the populated table to.
    additional_pip_requirements: Additional dependencies to be installed using
      pip to the pipeline environment.
    region: Region to run the batch job in. Default is us-west1.
    machine_type: The machine_type to use for the workers. Default is
      n2-custom-2-32768-ext.

  Returns:
    The PipelineResult of the pipeline, after the pipeline starts running.
  """
  saved_dataset = generated_dataset.GeneratedDataset.from_key(
      dataset_name, creation_time
  )
  if saved_dataset is None:
    raise ValueError(
        f'Cannot find a dataset matching {dataset_name} and {creation_time}'
    )
  pipeline = pipeline_utils.get_batch_pipeline(saved_dataset, output_path)
  requirements_fn = os.path.join(tempfile.mkdtemp(), 'requirements.txt')
  with open(requirements_fn, 'w') as f:  # pylint: disable=unspecified-encoding
    f.write('\n'.join(additional_pip_requirements))
  return _run_dataflow_pipeline(
      pipeline,
      get_batch_job_name(dataset_name),
      'dataflow',
      region=region,
      machine_type=machine_type,
      requirements_file=requirements_fn if additional_pip_requirements else None
  )


def run_streaming(
    num_custom_features: int
) -> beam.runners.runner.PipelineResult:
  """Runs a streaming pipeline that populates Examples with num_custom_features.

  Args:
    num_custom_features: The number of features to populate in the pipeline.

  Returns:
    The PipelineResult of the pipeline, after the pipeline starts running.
  """
  if num_custom_features <= 0:
    raise ValueError('num_custom_features must be positive!')
  pipeline = pipeline_utils.get_pubsub_streaming_pipeline(
      max_custom_features=num_custom_features,
  )
  return _run_dataflow_pipeline(
      pipeline,
      get_streaming_job_name(num_custom_features),
      'streaming',
      streaming=True,
      max_num_workers=_MAX_NUM_WORKERS,
      machine_type=f'n1-standard-{NUM_VCPUS_PER_WORKER}',
      experiments=[f'min_num_workers={_DEFAULT_MIN_NUM_WORKERS}'],
  )


def schedule_returning_to_default_min_workers(num_features: int) -> None:
  """Schedula a job to periodically return min_num_workers to the default.

  The job is scheduled to run every day in 4 PM UTC time. Please note that
  requests to change the number of workers right before the job runs, might not
  take effect, as it takes a few minutes for the worker to start, and changing
  the min_num_workers value before the request was acted upon MIGHT effectively
  cancel the reqeust.

  Args:
    num_features: Update the streaming job that takes care of examples with this
      many number of CustomFeatures.

  Raises:
    RuntimeError: In case the relevant job is not running.
  """
  job_id = get_streaming_job_id(num_features)
  if not job_id:
    raise RuntimeError(f'Streaming job for {num_features} is not running.')

  scheduling_job_name = f'restores-min-workers-{num_features}'
  gcloud_command = f'''/usr/bin/gcloud scheduler jobs delete \
    {scheduling_job_name} --location={DEFAULT_DATAFLOW_REGION} --quiet \
    2>/dev/null
  '''
  _run_command(gcloud_command)

  project_number = helpers.get_project_number()
  service_account = f'{project_number}-compute@developer.gserviceaccount.com'
  uri = _DATAFLOW_UPDATE_MIN_NUM_WORKERS_REST_API_FORMAT.format(
      job_id=job_id, project_id=helpers.get_project_id(),
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


def _get_current_min_job_workers(job_id: str) -> int:
  """Returns the current min_num_workers for the job given by job_id.

  This function assumes the job_id is running.

  Args:
    job_id: The job id of the requested job.
  """
  gcloud_command = f'''/usr/bin/gcloud dataflow jobs describe {job_id} \
    --region="{DEFAULT_DATAFLOW_REGION}" \
    --format="value(runtimeUpdatableParams.minNumWorkers)" 2>/dev/null'''
  result = _run_command(gcloud_command)
  return int(result.stdout)


def update_job_min_workers(
    num_features: int, *, num_workers_to_add: int
) -> int:
  """Updates the minimum number of workers of the relevant streaming job.

  Args:
    num_features: Update the streaming job that takes care of examples with this
      many number of CustomFeatures.
    num_workers_to_add: The number of workers to add to the minimum number. This
      number could be both positive or negative.

  Returns:
    The number of workers actually added to the job. This number might differ
    from num_workers_to_add due to minimum and maximum number of workers of the
    job.

  Raises:
    RuntimeError: In case the relevant job is not running.
  """
  job_id = get_streaming_job_id(num_features)
  if not job_id:
    raise RuntimeError(f'Streaming job for {num_features} is not running.')
  current_min_num_workers = _get_current_min_job_workers(job_id)
  new_min_num_workers = min(
      current_min_num_workers + num_workers_to_add, _MAX_NUM_WORKERS
  )
  new_min_num_workers = max(new_min_num_workers, _DEFAULT_MIN_NUM_WORKERS)
  gcloud_command = f'''/usr/bin/gcloud dataflow jobs update-options \
    --region="{DEFAULT_DATAFLOW_REGION}" \
    --min-num-workers={new_min_num_workers} {job_id} 2>/dev/null'''
  _run_command(gcloud_command)
  return new_min_num_workers - current_min_num_workers


def get_streaming_job_id(num_features: int) -> str:
  """Returns the id of the num_features population streaming job.

  Args:
    num_features: Update the streaming job that takes care of examples with this
      many number of CustomFeatures.
  """
  streaming_job_name = get_streaming_job_name(num_features)
  gcloud_command = f'''/usr/bin/gcloud dataflow jobs list \
    --filter="STATE:Running AND NAME:{streaming_job_name}" \
    --format="value(JOB_ID)" 2>/dev/null'''
  result = _run_command(gcloud_command)
  return result.stdout.rstrip()


def is_streaming_job_running(num_features: int) -> bool:
  """Returns whether the streaming job with num_features is running or not.

  Args:
    num_features: Update the streaming job that takes care of examples with this
      many number of CustomFeatures.
  """
  return bool(get_streaming_job_id(num_features))
