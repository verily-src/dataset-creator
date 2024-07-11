"""Implements the abstract CustomFeature.

A Feature is a type that holds metadata, and supports fetching the data upon
request.

"""

from __future__ import annotations

import abc
import hashlib
import json
from typing import Any, Mapping, Sequence, Union

import numpy as np
import tensorflow as tf

from dataset_creator.features import generic_typing
from dataset_creator.features import serializable

BasicPrimitiveType = Union[bool, int, float, bytes, str]
PrimitiveType = (
    Union[None, BasicPrimitiveType, Sequence[BasicPrimitiveType]]
)
BasicCompositeType = Union[np.ndarray, tf.Tensor]
CompositeType = Union[
    BasicCompositeType, Sequence[BasicCompositeType]
]
BasicValue = Union[BasicPrimitiveType, BasicCompositeType]
ValueFeature = Union[PrimitiveType, CompositeType]

_PRIMITIVE = 'None'


class CustomFeature(abc.ABC):
  """An abstract CustomFeature.

  CustomFeature is built to support computation in parallel, by inherently
  supporting splitting the feature to sub-values that can be processed
  independently and merged upon completion.
  In addition, CustomFeatures are expected to be entirely reconstructible using
  the get_config and from_config methods.

  A CustomFeature goes through 3 main phases in its life cycle:
  Initialize          Only holds the feature config
      |        <----  self.split, self.create_context, self.process, self.merge
   Populate           Returns the merged outputs
      |        <----  self.output_signatures
   Finalize           Filter out the merged outputs to get the final outputs

  Classes implementing the CustomFeature interface MUST implement the following:

    def split(self) -> Sequence[Any]: A method to split an instance to subvalues
      to be processed independently.

    def process(self, metadata_value: Any, context: Any) -> Any: A method to
      process each independent subvalue. Must return None upon failure. Callers
      are expected to drop the feature if one of the subvalues fails and NOT to
      pass None values to merge.

    def merge(self, values: Sequence[Any]) -> Mapping[str, ValueFeature]: A
      method to merge the different independent outputs of process to a single
      output mapping.

    @property
    def output_signature(self) -> Mapping[str, type[ValueFeature]]: A property
      to identify which of the merged outputs should be included in the final
      feature version.

  Optionally, classes can also implement:

    _serializable_classes: A class attribute that specifies the Serializable
      classes that are expected to be passed as parameters to this feature.
      This is a MUST so we can convert name+serialized to the original parameter
      while reconstructing the feature, so it is necessary for features that get
      Serializable objects as parameters.

    def create_context(self) -> Any: A method to create the context for process
      calls. This can be useful when a context is required for processing, but
      creating it might be an expensive operation. The context created by create
      context can be shared between features, thus optimizing the process calls.

  Please note that this class is picklable as long as the kwargs passed during
  initialization are picklable. The container attribute is not pickled and
  therefore must be reset upon unpickling.

  Attributes:
    container: An object containing this feature. This needs to be set
      explicitly upon adding the feature to the container.
    drop_on_finalize: Do not include any of the merged outputs in the finalized
      version of this feature. This is especially useful if you only want this
      feature as an intermediate to another feature. For example you only want
      images feature so you can do inference, but you're not interested in the
      images themselves.
  """

  # Since every CustomFeature must be reconstructible, we must keep track of all
  # possible Serializable classes that can be inputs to this CustomFeature, so
  # we know which one to call upon reconstruction.
  _serializable_classes: Sequence[type[serializable.Serializable]] = ()

  def __init__(self, drop_on_finalize: bool = False, **kwargs):
    """Initializes a CustomFeature instance."""

    kwargs['drop_on_finalize'] = drop_on_finalize
    for k, v in kwargs.items():
      if not generic_typing.generic_isinstance(v, PrimitiveType):
        if not isinstance(v, serializable.Serializable):
          raise ValueError(f'Value {v} of key {k} is invalid.')
        elif v.__class__ not in self._serializable_classes:
          raise ValueError(f'Value {v} of key {k} cannot be reconstructed.')
    self._kwargs: Mapping[str, Any] = kwargs
    self.drop_on_finalize = drop_on_finalize
    self.container: Container = {}

  @abc.abstractmethod
  def split(self) -> Sequence[Any]:
    """Splits the feature into values to be processed.

    Returns:
      A sequence made of single metadata values, to be processed by
      self._process.
    """

  def create_context(self) -> Any:
    """Precomputes an optional value that can be used in implementing process().

    This is helpful if you need a potentially expensive resource to compute a
    feature, e.g. a ML model. The context should be applicable to all features;
    a video file object would be a bad example even though it would be speed up
    the creation of many examples, it wouldn't be relevant for all examples
    (unless the entire dataset is a single video).

    Returns:
      A context to be provided to process calls. The context is expected to be
      shared between features as a means of optimization (instead of creating
      the context inside process every time).
    """
    return None

  @abc.abstractmethod
  def process(self, metadata_value: Any, context: Any) -> Any:
    """Processes a single output of self.split() into data for this feature.

    Please note that this method might be called for several outputs of
    self.split() in parallel, so it must be thread-safe.

    Args:
      metadata_value: The value to be processed.
      context: The context needed for processing. For example, a Keras model
        that is loaded in an outer scope. This is the result of a create_context
        call.

    Returns:
      The processed value, or None upon failure.
    """

  @abc.abstractmethod
  def merge(self, values: Sequence[Any]) -> Mapping[str, ValueFeature]:
    """Merges data values into a {name: feature_value} mapping.

    Items that have a matching key in the result of `output_signature()` will be
    serialized in the final example. Items that do not have a matching key can
    still be provided in order to help the processing of other features.

    For example, an ImagesFeature can produce encoded and decoded versions of
    images in merge(). The output_signature() may only specify the key for the
    encoded version so that it only stores compressed bytes. However an edge
    detection feature may use the already decoded image to detect edges.

    Args:
      values: The results from each of the process() calls, in the order
        provided by split().

    Returns:
      A mapping of {name: feature_value} containing the merged values.
    """

  @property
  @abc.abstractmethod
  def output_signature(self) -> Mapping[str, type[ValueFeature]]:
    """Returns the output signature of the merge method."""

  def get_config(self) -> FeatureConfig:
    """A serializable representation of this feature.

    Returns:
      A mapping with the format {kwarg_name@class_name: value}.
    """

    metadata = {}
    for k, v in self._kwargs.items():
      if generic_typing.generic_isinstance(v, PrimitiveType):
        metadata[f'{k}@{_PRIMITIVE}'] = v
      else:
        metadata[f'{k}@{v.__class__.__name__}'] = v.serialize()
    return metadata

  @classmethod
  def from_config(cls, config: FeatureConfig) -> Any:
    """Converts the metadata of a feature to a feature instance.

    Args:
      config: The metadata to convert. Keys must be 'kwarg_name@class_name'.

    Returns:
      A CustomFeature instance defined using the metadata.
    """

    kwargs = {}
    class_name_to_class = {
        serializable_class.__name__: serializable_class
        for serializable_class in cls._serializable_classes
    }
    for key, value in config.items():
      name, class_name = key.split('@')
      if class_name != _PRIMITIVE:
        if class_name not in class_name_to_class:
          raise ValueError(
              f'Found class name {class_name} which does not appear in'
              ' _serializable_classes.'
          )
        assert isinstance(value, bytes)
        # pylint: disable-next=line-too-long
        value = class_name_to_class[class_name].deserialize(value)  # type: ignore[assignment]
      kwargs[name] = value

    return cls(**kwargs)  # type: ignore[arg-type]

  @property
  def is_self_contained(self):
    """Whether this feature relies on other features for proper operation."""
    return True

  def __reduce__(self):
    return self.__class__.from_config, (self.get_config(),)

  def __hash__(self) -> int:
    config = self.get_config()
    sorted_config = {}
    for k in sorted(config):
      value = config[k]
      if isinstance(value, bytes):
        value = value.decode()
      sorted_config[k] = value
    serialized = json.dumps(sorted_config, sort_keys=True).encode()
    return int(hashlib.md5(serialized).hexdigest(), 16)

  def __eq__(self, value: object) -> bool:
    if not isinstance(value, self.__class__):
      return False
    return self.get_config() == value.get_config()


Feature = Union[ValueFeature, CustomFeature]
FeatureConfig = dict[str, PrimitiveType]
Container = Mapping[str, Feature]
