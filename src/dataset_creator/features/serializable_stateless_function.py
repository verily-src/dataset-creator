"""Implements SerializableStatelessFunction class.

Passing a python function between processes presents a challenge of validating
that the function does not rely on some local / global python context, as that
context will not be the same in the target process.
The main functionality implemented in SerializableStatelessFunction is
validation that the given function is indeed stateless.

Please additionaly note that using serializable python functions might have
security implications, so use this module when absolutely necessary and with
care.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
from typing import Any, Callable, Sequence

import dill  # type: ignore[import]

from dataset_creator.features import serializable


def _get_fn_from_serialized(serialized_code: bytes) -> Callable[..., Any]:
  return types.FunctionType(
      dill.loads(serialized_code),
      globals={'__builtins__': globals()['__builtins__']},
  )


@dataclasses.dataclass(frozen=True)
class SerializableStatelessFunction(serializable.Serializable):
  """Implements a serializable python function and validates it is stateless."""
  fn: Callable[..., Any]

  def validate(self, mock_args: Sequence[Any] = ()):
    """Validates that self.fn is indeed a stateless function.

    The validation is performed by running the function in a python environment
    that contains no globals (other than the function and its arguments.).
    Args:
      mock_args: Arguments to be passed to a simulation of the function.

    Raises:
      ValueError: In any case the function simulation raises an exception.
    """
    # inspect.isfunction must get a local variable.
    func = self.fn
    if not inspect.isfunction(func):
      raise ValueError('The input MUST be a function (not just a callable)')
    # We need to be subtle here. The function is stateless, but the arguments
    # might not be, so we need to make sure we completely detach the function
    # from the current context by serializing-then-deserializing.
    # This way, even if an argument that is passed to exec makes some modules
    # join along, the new deserialized function won't recognize it.
    func = _get_fn_from_serialized(self.serialize())
    local_context = {'fn': func, 'mock_arguments': mock_args}
    try:
      exec('fn(*mock_arguments)', {}, local_context)  # pylint: disable=exec-used
    except Exception as e:
      raise ValueError('Function is not stateless, raised on mock run.') from e

  def serialize(self) -> bytes:
    """Returns the serialized."""
    return dill.dumps(self.fn.__code__)

  @classmethod
  def deserialize(cls, serialized: bytes) -> SerializableStatelessFunction:
    return cls(_get_fn_from_serialized(serialized))

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)
