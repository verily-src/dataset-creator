"""Base environment for dataset creator."""

import dataclasses

from dataset_creator.environments import base_database
from dataset_creator.environments import base_pipeline_runner


@dataclasses.dataclass(frozen=True)
class BaseEnvironment:
  database: base_database.BaseDatabase
  pipeline_runner: base_pipeline_runner.BasePipelineRunner
  message_sender: base_pipeline_runner.BasePipelineMessagesHandler
