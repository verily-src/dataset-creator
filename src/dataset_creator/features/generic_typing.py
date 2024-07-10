"""Implements an isinstance function for typing types."""

import typing
from typing import Any, Sequence

import more_itertools
import numpy as np


def generic_isinstance(value: Any, typing_type: Any) -> bool:
  """A parallel of isinstance also supporting some typing types.

  Args:
    value: The value to check its type.
    typing_type: Check if value is an instance of this type.

  Returns:
    True if value is an instance of typing_type, False otherwise.
  """
  origin = typing.get_origin(typing_type)
  typing_arguments = typing.get_args(typing_type)
  if origin is None:
    return isinstance(value, typing_type)
  if origin == typing.Union:
    return any(
        generic_isinstance(value, underlying_type)
        for underlying_type in typing_arguments
    )
  if issubclass(origin, Sequence):
    if not generic_isinstance(value, origin):
      return False
    if not typing_arguments:
      return True
    if issubclass(origin, tuple):
      if Ellipsis in typing_arguments:
        if len(typing_arguments) != 2 or typing_arguments[0] == Ellipsis:
          raise ValueError(f'Unknown ellipsis annotation in {typing_type}')
        typing_arguments = typing_arguments[:1] * len(value)
      if len(typing_arguments) == len(value):
        return all(
            generic_isinstance(value[i], typing_arguments[i])
            for i in range(len(value))
        )
      return False
    if len(typing_arguments) == 1:
      return all(
          generic_isinstance(v, more_itertools.one(typing_arguments))
          for v in value
      )
  if origin == np.ndarray:
    if not isinstance(value, np.ndarray):
      return False
    return value.dtype.type is typing.get_args(typing_arguments[1])[0]
  raise ValueError(f'Type {typing_type} is not supported.')
