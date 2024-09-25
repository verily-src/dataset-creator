"""A module implementing the GCP environment for dataset creator."""

from dataset_creator.environments import base_environment
from dataset_creator.environments.gcp import gcp_database
from dataset_creator.environments.gcp import gcp_pipeline_runner


class GcpEnvironment(base_environment.BaseEnvironment):
  """An environment using spanner as the datatbase, dataflow and pubsub."""

  def __init__(self):
    super().__init__(
        database=gcp_database.SpannerDatabase(),
        pipeline_runner=gcp_pipeline_runner.DataflowPipelineRunner(),
        message_sender=gcp_pipeline_runner.PubsubMessagesHandler(),
    )
