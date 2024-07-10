"""Defines pytest fixtures."""

import collections
import os.path

import apache_beam as beam
from apache_beam.runners.direct import direct_runner
from apache_beam.runners.portability import fn_api_runner
import pytest
import requests


class _CustomDirectPipelineResult(direct_runner.DirectPipelineResult):
  def job_id(self) -> str:
    return 'test'


class _CustomFnRunnerResult(fn_api_runner.fn_runner.RunnerResult):
  def job_id(self) -> str:
    return 'test'


@pytest.fixture(scope='session', autouse=True)
def replace_dataflow_runner_with_direct_runner():
  beam.runners.dataflow.dataflow_runner.DataflowRunner = (
      direct_runner.DirectRunner
  )
  fn_api_runner.fn_runner.RunnerResult = _CustomFnRunnerResult
  direct_runner.DirectPipelineResult = _CustomDirectPipelineResult


@pytest.fixture(scope='session', autouse=True)
def replace_requests_get_when_trying_to_determine_project_id():
  def replaced_get(url, *_, **_2):
    if 'numeric' in url:
      value = '12345678'
    else:
      value = 'test-project-id'
    return collections.namedtuple('Response', ['text'])(value)

  requests.get = replaced_get


@pytest.fixture(scope='session', autouse=True)
def override_exists():
  original_exists = os.path.exists
  os.path.exists = lambda p: original_exists(p) or str(p).startswith('/gcs/')
