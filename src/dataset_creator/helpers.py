"""Defines types, classes used by components of dataset_creator."""

from __future__ import annotations

import functools
import glob
from typing import Callable, TypeVar

from google.cloud import pubsub_v1  # type: ignore[attr-defined]
from google.cloud import spanner  # type: ignore[attr-defined]
from google.cloud.spanner_v1 import database  # type: ignore[import]
import requests

T = TypeVar('T')
Supplier = Callable[[], T]

_SPANNER_INSTANCE_NAME = 'datasetcreator'
_SPANNER_DB_NAME = 'dataset_creator_db'

# pylint: disable=line-too-long
_GOOGLE_API_URL_FOR_PROJECT_ID = 'http://metadata.google.internal/computeMetadata/v1/project/project-id'
_GOOGLE_API_URL_FOR_PROJECT_NUMBER = 'http://metadata.google.internal/computeMetadata/v1/project/numeric-project-id'
# pylint: enable=line-too-long


def glob_prefix(prefix: str) -> list[str]:
  if not prefix.endswith('*'):
    prefix += '*'
  return glob.glob(prefix)


def get_db() -> database.Database:
  spanner_client = spanner.Client(project=get_project_id())
  instance = spanner_client.instance(_SPANNER_INSTANCE_NAME)
  return instance.database(_SPANNER_DB_NAME)


@functools.lru_cache()
def get_project_id() -> str:
  return requests.get(
      _GOOGLE_API_URL_FOR_PROJECT_ID,
      headers={'Metadata-Flavor': 'Google'},
      timeout=60,
  ).text


@functools.lru_cache()
def get_project_number() -> str:
  return requests.get(
      _GOOGLE_API_URL_FOR_PROJECT_NUMBER,
      headers={'Metadata-Flavor': 'Google'},
      timeout=60,
  ).text


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
  project_id = get_project_id()
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


def get_project_bucket_path() -> str:
  return f'/gcs/{get_project_id()}-dataset-creator'
