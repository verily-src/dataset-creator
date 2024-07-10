"""Tests for dataflow_utils."""

import collections
import datetime
import io
import re
import subprocess
import tempfile
from unittest import mock

from absl.testing import parameterized  # type: ignore[import]
import apache_beam as beam
import more_itertools
import pytest

from dataset_creator import generated_dataset
from dataset_creator import test_utils
from dataset_creator.pipeline import dataflow_utils

# pylint: disable=protected-access

_StandardOptions = beam.options.pipeline_options.StandardOptions


@pytest.fixture(scope='function')
def mock_gcloud_and_streaming_pipeline_running():
  original_subprocess_run = subprocess.run
  current_min_num_workers = dataflow_utils._DEFAULT_MIN_NUM_WORKERS

  def mock_subprocess_run(cmd: str, **_):
    nonlocal current_min_num_workers
    CmdOutput = collections.namedtuple('CmdOutput', ['stdout'])
    if 'minNumWorkers' in cmd:
      # This is an attempt to read the current min_num_workers.
      return CmdOutput(str(current_min_num_workers))
    if 'STATE:Running' in cmd:
      # An attempt to get the job_id.
      return CmdOutput('test')
    if 'gcloud scheduler jobs' in cmd:
      return
    # Else, we mock the operation of adding to the min_num_workers.
    assert 'update-options' in cmd
    num_workers = more_itertools.one(re.findall(r'min-num-workers=(\d+)', cmd))
    current_min_num_workers = int(num_workers)

  subprocess.run = mock_subprocess_run
  yield
  subprocess.run = original_subprocess_run


class DataflowUtilsTest(parameterized.TestCase):
  def test_get_batch_job_name_returns_a_valid_job_name(self):
    dataflow_utils.validate_job_name(dataflow_utils.get_batch_job_name('test'))

  def test_get_streaming_job_name_returns_a_valid_job_name(self):
    dataflow_utils.validate_job_name(dataflow_utils.get_streaming_job_name(2))

  def test_validate_job_raises_valueerror_on_invalid_name(self):
    with self.assertRaisesRegex(ValueError, 'Name must conform to'):
      dataflow_utils.validate_job_name('my_test')

  def test_run_streaming_pipeline(self):
    result = dataflow_utils.run_streaming(num_custom_features=1)
    self.assertEqual(result.state, 'RUNNING')
    pipeline_options = result._evaluation_context.pipeline_options
    self.assertEqual(
        pipeline_options.view_as(_StandardOptions).streaming, True
    )

  @parameterized.named_parameters(
      ('zero', 0), ('negative', -1)
  )
  def test_run_streaming_pipeline_raises_with_invalid_num_features(
      self, num_features: int
  ):
    with self.assertRaisesRegex(ValueError, 'must be positive!'):
      dataflow_utils.run_streaming(num_custom_features=num_features)

  def test_is_streaming_pipeline_running(self):
    self.assertFalse(dataflow_utils.is_streaming_job_running(1))

  def _run_batch_pipeline(self) -> beam.runners.runner.PipelineResult:
    dataset_name = 'dataset_creator_test'
    creation_time = datetime.datetime.now()
    generated_dataset.GeneratedDataset(
        dataset_name=dataset_name,
        creation_time=creation_time,
        dataset_creator_version='test',
        get_examples_generator=test_utils.get_examples_generator,
    ).save()
    with tempfile.NamedTemporaryFile() as tmpfile:
      return dataflow_utils.run_batch(dataset_name, creation_time, tmpfile.name)

  @mock.patch('sys.stdout', new_callable=io.StringIO)
  def test_run_batch_prints_link_to_job(self, mock_stdout):
    self._run_batch_pipeline()
    self.assertIn('https://', mock_stdout.getvalue())

  def test_run_batch(self):
    result = self._run_batch_pipeline()
    self.assertIn(result.state, ['RUNNING', 'DONE'])

  def test_run_batch_raises_with_invalid_dataset_params(self):
    with self.assertRaisesRegex(ValueError, 'Cannot find a dataset'):
      dataflow_utils.run_batch('test', datetime.datetime.now(), '/tmp/test')

  def test_update_job_raises_when_job_not_running(self):
    # Notice we don't use the fixture that simulates running in this test.
    with self.assertRaisesRegex(RuntimeError, 'is not running'):
      dataflow_utils.update_job_min_workers(1, num_workers_to_add=1)

  @pytest.mark.usefixtures('mock_gcloud_and_streaming_pipeline_running')
  def test_update_job_returns_num_workers_increased(self):
    extra_worker_capacity = (
        dataflow_utils._MAX_NUM_WORKERS
        -
        dataflow_utils._DEFAULT_MIN_NUM_WORKERS
    )
    added_workers = dataflow_utils.update_job_min_workers(
        1, num_workers_to_add=5
    )
    self.assertEqual(added_workers, 5)
    extra_worker_capacity -= added_workers
    added_workers = dataflow_utils.update_job_min_workers(
        1, num_workers_to_add=extra_worker_capacity + 1000
    )
    self.assertEqual(added_workers, extra_worker_capacity)
    # Now workers are at full capacity, try to reduce the number by too much
    dataflow_utils.update_job_min_workers(1, num_workers_to_add=int(-1e6))
    self.assertEqual(
        dataflow_utils._get_current_min_job_workers('test'),
        dataflow_utils._DEFAULT_MIN_NUM_WORKERS
    )

  @pytest.mark.usefixtures('mock_gcloud_and_streaming_pipeline_running')
  def test_schedule_returning_to_default_min_workers(self):
    dataflow_utils.schedule_returning_to_default_min_workers(1)

  def test_schedule_raises_when_job_is_not_running(self):
    # Notice we don't use the fixture that simulates running in this test.
    with self.assertRaisesRegex(RuntimeError, 'is not running'):
      dataflow_utils.schedule_returning_to_default_min_workers(1)

# pylint: enable=protected-access
