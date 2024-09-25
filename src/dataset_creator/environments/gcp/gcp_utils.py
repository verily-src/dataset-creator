"""A module for common GCP utilities."""

import functools

import requests

# pylint: disable=line-too-long
_GOOGLE_API_URL_FOR_PROJECT_ID = 'http://metadata.google.internal/computeMetadata/v1/project/project-id'
_GOOGLE_API_URL_FOR_PROJECT_NUMBER = 'http://metadata.google.internal/computeMetadata/v1/project/numeric-project-id'
# pylint: enable=line-too-long


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
