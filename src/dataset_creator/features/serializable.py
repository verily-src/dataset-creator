"""Defines the interface for serializable objects."""

from __future__ import annotations

import abc


class Serializable(abc.ABC):
  """Interface for serializable objects."""

  @abc.abstractmethod
  def serialize(self) -> bytes:
    """Returns the serialization of this object."""

  @classmethod
  @abc.abstractmethod
  def deserialize(cls, serialized: bytes) -> Serializable:
    """Returns the deserialized object."""

  def __reduce__(self):
    return self.__class__.deserialize, (self.serialize(),)

