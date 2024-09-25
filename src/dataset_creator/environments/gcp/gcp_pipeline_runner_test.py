"""Tests for gcp_pipeline_runner.py."""

import collections
import re
import subprocess

from absl.testing import parameterized  # type: ignore[import]
import apache_beam as beam
import more_itertools
import pytest

from dataset_creator.environments.gcp import gcp_pipeline_runner

# pylint: disable=protected-access

_StandardOptions = beam.options.pipeline_options.StandardOptions


@pytest.fixture(scope='function')
def mock_gcloud_and_streaming_pipeline_running():
  original_subprocess_run = subprocess.run
  current_min_num_workers = gcp_pipeline_runner._DEFAULT_MIN_NUM_WORKERS

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


class DataflowPipelineRunnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dataflow_runner = gcp_pipeline_runner.DataflowPipelineRunner()

  @parameterized.named_parameters(('batch', True), ('streaming', False))
  def test_get_job_name_returns_a_valid_name(self, is_batch: bool):
    if is_batch:
      get_job_name_fn = self.dataflow_runner.get_batch_job_name
    else:
      get_job_name_fn = self.dataflow_runner.get_streaming_job_name
    self.dataflow_runner.validate_job_name(get_job_name_fn('test'))

  def test_validate_job_name_raises_with_invalid_job_name(self):
    with self.assertRaisesRegex(ValueError, 'Invalid name! Name must conform'):
      self.dataflow_runner.validate_job_name('name_with_underscores')

  # def test_run_streaming_pipeline(self):
  #   pubsub_messages_handler = gcp_pip
  #   pipeline = lambda root: (root | )
  #   result = self.dataflow_runner.run_streaming(basename='1',)
  #   self.assertEqual(result.state, 'RUNNING')
  #   pipeline_options = result._evaluation_context.pipeline_options
  #   self.assertEqual(
  #       pipeline_options.view_as(_StandardOptions).streaming, True
  #   )

# pylint: enable=protected-access
