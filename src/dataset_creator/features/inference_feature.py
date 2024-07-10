"""A module for adding feature vectors from Keras models."""

import functools
from typing import Any, Callable, Mapping, Sequence, Union

import more_itertools
import tensorflow as tf
from typing_extensions import TypeAlias

from dataset_creator.features import base_feature
from dataset_creator.features import inference_model
from dataset_creator.features import serializable_stateless_function

_SerializableStatelessFunction = (
    serializable_stateless_function.SerializableStatelessFunction
)
_ValueFeature = base_feature.ValueFeature
_Container = base_feature.Container

_ContainerToInputs = Callable[[_Container], Sequence[_ValueFeature]]
_InferenceOutput = Mapping[str, tf.Tensor]

_SerializableStatelessFnType: TypeAlias = (
    serializable_stateless_function.SerializableStatelessFunction
)


@functools.lru_cache()
def _get_inference_model(
    keras_model_path: str,
    outputs_layer_names: Sequence[str]
) -> tf.keras.Model:
  return inference_model.inference_model(
      tf.keras.models.load_model(keras_model_path), outputs_layer_names
  )


class InferenceFeature(base_feature.CustomFeature):
  """A feature of Keras model feature vectors."""

  _serializable_classes = [_SerializableStatelessFunction]

  def __init__(
      self,
      keras_model_path: str,
      outputs_layer_names: Sequence[str],
      container_to_inputs: Union[
          _SerializableStatelessFnType, _ContainerToInputs
      ],
      **kwargs
  ):
    """Instantiate an InferenceFeature.

    Args:
      keras_model_path: A path to a Keras model to use for inference.
      outputs_layer_names: The names of layers whose outputs are extracted in
        inference.
      container_to_inputs: A callable that receives self.container as an input
        and returns a Sequence of model inputs. Note that this callable must be
        completely stateless.
      **kwargs: Additional keyword arguments to be passed to CustomFeature.
    """
    if not isinstance(container_to_inputs, _SerializableStatelessFunction):
      container_to_inputs = _SerializableStatelessFunction(container_to_inputs)

    super().__init__(
        keras_model_path=keras_model_path,
        outputs_layer_names=outputs_layer_names,
        container_to_inputs=container_to_inputs,
        **kwargs
    )

    self._keras_model_path = keras_model_path
    self._outputs_layer_names = outputs_layer_names
    self._container_to_inputs: _SerializableStatelessFnType = (
        container_to_inputs
    )

  def split(self) -> Sequence[_ValueFeature]:
    """Returns the different inputs of the model for this feature."""
    return self._container_to_inputs(self.container)

  def create_context(self) -> tf.keras.Model:
    """Creates the inference model as a shared context.

    Returns:
      An inference model whose output's format is {layer_name: layer_output}.
    """
    # Validate that the given callable is indeed stateless when we create the
    # context, before we do any specific processing.
    self._container_to_inputs.validate([self.container])
    if not isinstance(self._container_to_inputs(self.container), Sequence):
      raise ValueError('Returned inputs from the callback must be a sequence.')
    return _get_inference_model(
        self._keras_model_path, tuple(self._outputs_layer_names)
    )

  def process(
      self, metadata_value: base_feature.BasicCompositeType,
      context: tf.keras.Model
  ) -> _InferenceOutput:
    """Returns the model evaluation on model_input.

    Args:
      metadata_value: The input to evaluate.
      context: The model to use for evaluation.
    """
    return context(metadata_value, training=False)

  def merge(self, values: Sequence[_InferenceOutput]) -> dict[str, tf.Tensor]:
    """Concatenates the results of different inferences.

    Args:
      values: Individual inference outputs.

    Returns:
      A mapping from layer_name to inference layer output.
    """
    model_outputs = {}
    layer_names = more_itertools.first(values).keys()
    for layer_name in layer_names:
      model_outputs[layer_name] = tf.concat(
          [inference_value[layer_name] for inference_value in values], 0,
      )
    return model_outputs

  @property
  def output_signature(self) -> Mapping[str, Any]:
    return {layer_name: tf.Tensor for layer_name in self._outputs_layer_names}

  @property
  def is_self_contained(self):
    """See base class."""
    return False

