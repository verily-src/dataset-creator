"""Implements a GenericLambdaFeature."""

from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import numpy.typing  # pylint: disable=unused-import
import tensorflow as tf
from typing_extensions import TypeAlias

from dataset_creator.features import base_feature
from dataset_creator.features import generic_typing
from dataset_creator.features import serializable_stateless_function

_Context: TypeAlias = Any
_ValueFeature = base_feature.ValueFeature
_Container = base_feature.Container

_SerializableStatelessFnType: TypeAlias = (
    serializable_stateless_function.SerializableStatelessFunction
)
_SplitFn: TypeAlias = Union[
    _SerializableStatelessFnType, Callable[[_Container], Sequence[Any]]
]
_CreateContextFn: TypeAlias = Union[
  _SerializableStatelessFnType, Callable[[], Any]
]
_ProcessFn: TypeAlias = Union[
  _SerializableStatelessFnType, Callable[[Any, Any], Any], Callable[[Any], Any]
]
_MergeFn: TypeAlias = Union[
    _SerializableStatelessFnType,
    Callable[[Sequence[Any]], Mapping[str, _ValueFeature]]
]

_SerializableStatelessFn = (
    serializable_stateless_function.SerializableStatelessFunction
)


def _value_to_generic_type(value: _ValueFeature) -> type[_ValueFeature]:
  # Treat tensors explicitly to return a Tensor and not EagerTensor for example.
  if isinstance(value, tf.Tensor):
    return tf.Tensor
  if isinstance(value, np.ndarray):
    return np.typing.NDArray[value.dtype.type]  # type: ignore[name-defined]
  if generic_typing.generic_isinstance(
      value, Union[base_feature.BasicValue, type(None)]
  ):
    return type(value)
  assert isinstance(value, Sequence)
  if not value:
    return Sequence[float]
  return Sequence[_value_to_generic_type(value[0])]  # type: ignore[misc]


class GenericLambdaFeature(base_feature.CustomFeature):
  """A class for applying some stateless function on the container in parallel.

  In order to allow parallelization, the class requires the user to specify how
  to split each container to processing inputs, and how to merge the different
  processing outputs into a single output.
  """

  _serializable_classes = (_SerializableStatelessFn,)

  def __init__(
      self,
      split_fn: _SplitFn,
      process_fn: _ProcessFn,
      merge_fn: _MergeFn,
      create_context_fn: Optional[_CreateContextFn] = None,
      process_with_context: bool = True,
      **kwargs
  ):
    """Instantiates a GenericLambdaFeature.

    Args:
      split_fn: A stateless function which receives the example populated thus
        far, and returns the different process inputs.
      process_fn: A stateless function which receives A value from split and the
        context (assuming process_with_context), and returns the processed
        subvalue. If process_with_context is False, process_fn is expected to
        accept the split value only, without the context.
      merge_fn: A stateless function which receives the sequence of processed
        values (ordered according to the order of split) and returns a single
        Mapping[str, ValueFeature] that is the result of the feature processing.
      create_context_fn: A stateless function which creates the context for
        processing. Default is None, which creates a None context.
      process_with_context: If True, process_fn accepts 2 arguments (the value
        to be processed and the context). Else, process_fn accepts only 1
        argument (the value to be processed).
      **kwargs: Additional kwargs to pass to the CustomFeature.
    """
    create_context_fn = create_context_fn or (lambda: None)
    if not isinstance(split_fn, _SerializableStatelessFn):
      split_fn = _SerializableStatelessFn(split_fn)
    if not isinstance(process_fn, _SerializableStatelessFn):
      process_fn = _SerializableStatelessFn(process_fn)
    if not isinstance(merge_fn, _SerializableStatelessFn):
      merge_fn = _SerializableStatelessFn(merge_fn)
    if not isinstance(create_context_fn, _SerializableStatelessFn):
      create_context_fn = _SerializableStatelessFn(create_context_fn)

    super().__init__(
        split_fn=split_fn,
        process_fn=process_fn,
        merge_fn=merge_fn,
        create_context_fn=create_context_fn,
        process_with_context=process_with_context,
        **kwargs
    )

    self._split_fn: _SerializableStatelessFnType = split_fn
    self._process_fn: _SerializableStatelessFnType = process_fn
    self._merge_fn: _SerializableStatelessFnType = merge_fn
    self._create_context_fn: _SerializableStatelessFnType = create_context_fn
    self._process_with_context = process_with_context

  def create_context(self) -> Any:
    """See base class."""
    self._split_fn.validate([self.container])
    split = list(self._split_fn(self.container))
    if self._process_with_context:
      self._create_context_fn.validate([])
      context = self._create_context_fn()  # type: ignore[misc]
      self._process_fn.validate([split[0], context])
      single_processed = self._process_fn(
          split[0], context
      )  # type: ignore[call-arg]
    else:
      self._process_fn.validate([split[0]])
      single_processed = self._process_fn(split[0])  # type: ignore[call-arg]
      context = None
    self._merge_fn.validate([[single_processed] * len(split)])
    return context

  def split(self) -> Sequence[Any]:
    """See base class."""
    return self._split_fn(self.container)

  def process(self, metadata_value: Any, context: Any) -> Any:
    """See base class."""
    if self._process_with_context:
      return self._process_fn(metadata_value, context)  # type: ignore[call-arg]
    return self._process_fn(metadata_value)  # type: ignore[call-arg]

  def merge(self, values: Sequence[Any]) -> Mapping[str, _ValueFeature]:
    """See base class."""
    return self._merge_fn(values)

  @property
  def output_signature(self) -> Mapping[str, type[_ValueFeature]]:
    """See base class."""
    split = list(self._split_fn(self.container))
    if self._process_with_context:
      single_processed = self._process_fn(
          split[0], self._create_context_fn()
      )  # type: ignore[misc,call-arg]
    else:
      single_processed = self._process_fn(split[0])  # type: ignore[call-arg]
    merged = self._merge_fn([single_processed] * len(split))
    return {k: _value_to_generic_type(v) for k, v in merged.items()}

  @property
  def is_self_contained(self):
    """See base class."""
    return False
