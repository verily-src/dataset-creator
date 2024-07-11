"""Implements the conversion of a Keras Model to a Keras inference model.

Convert a tf.keras.Model to a new model whose outputs are the requested layers.
Multiple outputs are supported for a single inference model.

Usage:
  inference_model.inference_model(model, outputs_layer_names)
"""

import collections
from typing import Iterable, Mapping, Sequence

import more_itertools
import tensorflow as tf


def _model_layers_input_names(
    model: tf.keras.Model,
) -> Mapping[str, Sequence[str]]:
  """Returns a mapping layer.name -> names of input layers for that key.

  In case a layer has no inputs, the matching value is a Sequence of length 0.

  Args:
    model: The model to be analyzed.
  """
  input_names = collections.defaultdict(list)
  # model.layers sometimes does not contain the input layer, so iterate all
  # layers starting from the inputs using BFS.
  for layer in _bfs_model_layers(model):
    for node in layer.outbound_nodes:
      input_names[node.outbound_layer.name].append(layer.name)
  return input_names


# Required because of a bug, details in documentation or in b/267702457.
def inference_model(
    model: tf.keras.Model,
    outputs_layer_names: Sequence[str],
) -> tf.keras.Model:
  """Returns the converted model with Identity layers after output layers.

  The outputs of the returned Model have the following structure:
  {outputs_layer_name: layer_outputs}

  The naive approach would be something like:
  outputs = {layer.name: layer.output
             for layer in map(model.get_layer, outputs_layer_names)}
  return tf.keras.Model(full_model.inputs, outputs=outputs)

  The problem with the naive approach is in the case where we want to extract
  features from a Functional layer. In that case, building the model by asking
  for the Functional layer's output, makes the sublayers of the Functional layer
  to be expanded. In that case, layers that were not part of the original model
  by themselves are now part of the new model, and that created problems, since
  those layers are not connected by themselves to previous layers of the model.
  To deal with that issue, we would simply insert a tf.keras.layers.Identity
  layer after every Functional layer, so asking for THAT layer's output will not
  expand the Functional model.

  Args:
    model: The model to be converted.
    outputs_layer_names: The names of layers to output by the inference model.

  Returns:
    The inference model whose outputs are {outputs_layer_name: layer_outputs}.
  """

  input_names = _model_layers_input_names(model)
  inference_model_outputs = {}  # requested output name -> output_tensor
  layer_output_tensors = {}  # layer.name -> an output Tensor of that layer

  for layer in _bfs_model_layers(model):
    if not input_names[layer.name]:  # Input layer
      layer_output_tensors[layer.name] = layer.output
      continue

    input_tensors = [
        layer_output_tensors[input_layer_name]
        for input_layer_name in input_names[layer.name]
    ]
    if len(input_tensors) == 1:
      input_tensors = more_itertools.one(input_tensors)

    layer_output = layer(input_tensors)

    if layer.name in outputs_layer_names:
      layer_output = tf.keras.layers.Identity()(layer_output)
      inference_model_outputs[layer.name] = layer_output

    layer_output_tensors[layer.name] = layer_output

  missing_layers = set(outputs_layer_names) - set(layer_output_tensors.keys())
  if missing_layers:
    missing_layers_str = ', '.join(sorted(missing_layers))
    raise ValueError(
        f"{missing_layers_str} are not present in the model's layers"
    )

  return tf.keras.Model(inputs=model.inputs, outputs=inference_model_outputs)


def _bfs_model_layers(model: tf.keras.Model) -> Iterable[tf.keras.layers.Layer]:
  """Yields all the layers in the model ordered by depth (inputs first).

  The depth of a layer is its maximal distance from the model inputs. Please
  note that a layer might have several inputs, so finding a layer in BFS does
  not necessarily imply we have reached its maximal distance.

  The upside of using _bfs_model_layers, other than getting layers ordered by
  depth, over simply iterating over model.layers is that this way we ensure that
  the Input layers are always included. In Sequential models, for example,
  model.layers does not contain the Inputs.

  Args:
    model: The model to be iterate over.
  """
  next_layers = collections.deque(
      [model_input.node.layer for model_input in tf.nest.flatten(model.inputs)]
  )
  # We keep track of visited layers so we know when we've seen all the inputs
  # for a certain layer.
  visited_layers = set()
  while next_layers:
    layer = next_layers.popleft()
    yield layer
    visited_layers.add(layer)
    for node in layer.outbound_nodes:
      # We verify that we reached the maximal distance of node.layer from
      # model inputs by checking that all its inputs are visited.
      all_inputs_visited = True
      for input_node in node.layer.inbound_nodes:
        for input_layer in tf.nest.flatten(input_node.inbound_layers):
          if input_layer not in visited_layers:
            all_inputs_visited = False
            break
        if not all_inputs_visited:
          break
      if all_inputs_visited:
        next_layers.append(node.layer)
