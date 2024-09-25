"""A module defining base classes for running pipelines in the environment."""

import abc
from typing import Any, Callable, Iterable

import apache_beam as beam


class BasePipelineRunner(abc.ABC):
  """Base class for running batch and streaming pipelines."""

  @abc.abstractmethod
  def get_batch_job_name(self, basename: str) -> str:
    """Returns a name for a batch job."""

  @abc.abstractmethod
  def get_streaming_job_name(self, basename: str) -> str:
    """Returns a name for a streaming job."""

  @abc.abstractmethod
  def validate_job_name(self, job_name: str) -> None:
    """Validates a job name.

    Raises:
      ValueError: In case job name is invalid.
    """

  @abc.abstractmethod
  def run_batch(
      self,
      basename: str,
      pipeline: Callable[[beam.Pipeline], None],
      **pipeline_options: Any,
  ) -> beam.runners.runner.PipelineResult:
    """Runs a batch pipeline. Prints a link to the pipeline page.

    Args:
      basename: A basename to derive the job name from.
      pipeline: The pipeline to run.
      **pipeline_options: The pipeline options to use.
    """

  @abc.abstractmethod
  def run_streaming(
      self,
      basename: str,
      pipeline: Callable[[beam.Pipeline], None],
      **pipeline_options: Any,
  ) -> beam.runners.runner.PipelineResult:
    """Runs a streaming pipeline. Prints a link to the pipeline page.

    Args:
      basename: A basename to derive the job name from.
      pipeline: The pipeline to run.
      **pipeline_options: The pipeline options to use.
    """

  @abc.abstractmethod
  def is_job_running(self, job_name: str) -> bool:
    """Checks if a job is running."""

  @abc.abstractmethod
  def update_job_min_num_workers(
      self, job_name: str, *, num_workers_to_add: int,
  ) -> int:
    """Updates the minimum number of workers in the streaming job."""


class BasePipelineMessagesHandler(abc.ABC):
  """Base class for sending / recving messages to / from streaming pipelines."""

  @abc.abstractmethod
  def send_messages_to_pipeline(
      self, messages: Iterable[bytes], **attributes: str
  ) -> None:
    """Sends messages to the pipeline's message queue.

    Args:
      messages: The messages to send.
      **attributes: Additional attributes to send with all messages.
    """

  @abc.abstractmethod
  def recv_messages_from_pipeline(
      self, **filter_attributes: str
  ) -> Iterable[bytes]:
    """Receives messages from the pipeline.

    Args:
      **filter_attributes: Attributes to filter messages by.

    Returns:
      An iterable of messages processed by the pipeline.
    """

  @abc.abstractmethod
  def create_read_messages_ptransform(
      self, **filter_attributes: str
  ) -> beam.PTransform:
    """Creates a PTransform to read messages from the input message queue."""

  @abc.abstractmethod
  def create_write_out_messages_ptransform(
      self, **attributes
  ) -> beam.PTransform:
    """Creates a PTransform to write messages to the output message queue."""
