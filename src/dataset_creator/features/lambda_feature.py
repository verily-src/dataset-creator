"""Implements a LambdaFeature."""

from typing import Callable, Mapping, Union

from typing_extensions import TypeAlias

from dataset_creator.features import base_feature
from dataset_creator.features import generic_lambda_feature
from dataset_creator.features import serializable_stateless_function

_ValueFeature = base_feature.ValueFeature
_Container = base_feature.Container
_Function = Callable[[_Container], Mapping[str, _ValueFeature]]
_SerializableStatelessFnType: TypeAlias = (
    serializable_stateless_function.SerializableStatelessFunction
)
_SerializableStatelessFunction = (
    serializable_stateless_function.SerializableStatelessFunction
)


class LambdaFeature(generic_lambda_feature.GenericLambdaFeature):
  """A class for applying some stateless function on the container.

  Please note that the calculation of the given function on the container is not
  parallelized in any way.
  """

  def __init__(
      self,
      lambda_fn: Union[_SerializableStatelessFnType, _Function, None] = None,
      **kwargs
  ):
    if lambda_fn is None:
      if 'process_fn' not in kwargs:
        raise ValueError('lambda_fn must be provided.')
      # When we recreate this feature from its config, the config only holds
      # the GenericLambdaFeature values, so extract it this way.
      lambda_fn = kwargs['process_fn']
    if not isinstance(lambda_fn, _SerializableStatelessFunction):
      lambda_fn = _SerializableStatelessFunction(lambda_fn)

    # This feature could have been reconstructed with GenericLambdaFeature
    # keyword arguments. Remove them as we intend to pass them explicitly.
    kwargs.pop('split_fn', None)
    kwargs.pop('process_fn', None)
    kwargs.pop('merge_fn', None)
    kwargs.pop('process_with_context', None)

    super().__init__(
        split_fn=lambda container: [container],
        process_fn=lambda_fn,
        merge_fn=lambda values: values[0],
        process_with_context=False,
        **kwargs
    )
