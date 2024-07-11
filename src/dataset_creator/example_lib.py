"""Implements an Example, which is an ordered mapping feature_name->feature."""

from __future__ import annotations

import base64
import collections
import functools
import inspect
import typing
from typing import Any, Mapping, Sequence, Union
import zlib

import immutabledict  # type: ignore[import]
import more_itertools
import numpy as np
import numpy.typing  # pylint: disable=unused-import
import tensorflow as tf

from dataset_creator.features import base_feature
from dataset_creator.features import generic_lambda_feature
from dataset_creator.features import generic_typing
from dataset_creator.features import images_feature
from dataset_creator.features import inference_feature
from dataset_creator.features import lambda_feature
from dataset_creator.features import multi_lead_signal_feature
from dataset_creator.features import serializable

_BasicPrimitiveType = base_feature.BasicPrimitiveType
_PrimitiveType = base_feature.PrimitiveType
_CustomFeature = base_feature.CustomFeature
_FeatureConfig = base_feature.FeatureConfig
_ValueFeature = base_feature.ValueFeature

_FEATURE_TYPE = '_FEATURE_TYPE'
_FEATURE_SECONDARY_TYPE = '_FEATURE_SECONDARY_TYPE'
_INDEX = '_FEATURE_INDEX'
_PLACEHOLDER = '_PLACEHOLDER'
_TF_EXAMPLE_SEPARATOR_CHAR = '$'
_NESTING_CHAR = '/'

_TYPE_NAME_TO_TYPE: dict[str, type[base_feature.Feature]] = {
    'NoneType': type(None),
    'str': str,
    'bytes': bytes,
    'int': int,
    'float': float,
    'bool': bool,
    'ndarray': np.ndarray,
    'EagerTensor': tf.Tensor,
    'list': list,  # type: ignore[dict-item]
    'tuple': tuple,  # type: ignore[dict-item]
    'ImagesFeature': images_feature.ImagesFeature,
    'InferenceFeature': inference_feature.InferenceFeature,
    'MultiLeadSignalFeature': multi_lead_signal_feature.MultiLeadSignalFeature,
    'LambdaFeature': lambda_feature.LambdaFeature,
    'GenericLambdaFeature': generic_lambda_feature.GenericLambdaFeature,
}

_PYTHON_TO_TF_FEATURE_TYPE = {
    bool: tf.int64,
    str: tf.string,
    bytes: tf.string,
    float: tf.float32,
    int: tf.int64,
}

_STRING_TO_NUMPY_TYPE = {
    'float': np.float32,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'int': np.int32,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'bool': np.bool_,
    'bool_': np.bool_,
}

_MAX_LINE_LENGTH = 80

_generic_isinstance = generic_typing.generic_isinstance


@tf.function
def normalize_parsed_tf_example(
    parsed_tf_example: dict[str, Union[tf.Tensor, tf.SparseTensor]]
) -> dict[str, tf.Tensor]:
  """Normalizes a parsed tf.train.Example.

  Args:
    parsed_tf_example: A mapping returned from parsing a serialized version of
      Example.to_tf_example() by using tf.io.parse_single_example or any similar
      function.

  Returns:
    A normalized mapping after correcting and discarding irrelevant parts of it:
      1. SparseTensors are converted to Tensors.
      2. Tensors are being casted to tf.bool based on the type information
         encoded in their key.
      3. Numpy arrays are reshaped to their correct shape.
  """
  normalized = {}
  for key, tensor in parsed_tf_example.items():
    name, feature_key, _, inner_type = key.split(_TF_EXAMPLE_SEPARATOR_CHAR)

    normal_tensor = _normalize_tensor(tensor, inner_type)
    if feature_key == _PLACEHOLDER:
      normalized[name] = normal_tensor
    elif feature_key.startswith('serialized_bytes_'):
      decompressed = tf.io.decode_compressed(
          normal_tensor, compression_type='ZLIB'
      )
      normalized[name] = tf.io.parse_tensor(decompressed, out_type=np.string_)
    elif feature_key.startswith('serialized_'):
      decompressed = tf.io.decode_compressed(
          normal_tensor, compression_type='ZLIB'
      )
      dtype_key = _TF_EXAMPLE_SEPARATOR_CHAR.join(
          [name, feature_key.replace('serialized_', 'dtype_'), 'str', 'str']
      )
      dtype_tensor = _normalize_tensor(parsed_tf_example[dtype_key], 'str')

      def parse_tensor(dtype, tensor):
        return tf.cast(
            tf.io.parse_tensor(tensor, out_type=dtype), tf.float32
        )

      pred_fn_pairs = []
      for string_dtype, dtype in _STRING_TO_NUMPY_TYPE.items():
        pred_fn_pairs.append(
            (
                tf.equal(string_dtype, dtype_tensor),
                functools.partial(parse_tensor, dtype, decompressed),
            )
        )
      normalized[name] = tf.case(pred_fn_pairs, exclusive=True)
  return normalized


@tf.function
def _normalize_tensor(
    tensor: Union[tf.Tensor, tf.SparseTensor], dtype: str,
) -> tf.Tensor:
  if isinstance(tensor, tf.SparseTensor):
    dense_tensor = tf.sparse.to_dense(tensor)
  else:
    dense_tensor = tensor
  if dtype == 'bool':
    return tf.cast(dense_tensor, tf.bool)
  return dense_tensor


def _get_mock_value(full_type: type[_ValueFeature]) -> _ValueFeature:
  """Creates a mock value for a given type.

  Args:
    full_type: The type to create a value for. Example of given types could be
    'Sequence[int]' or 'np.ndarray'.

  Returns:
    The mock value.
  """
  origin = typing.get_origin(full_type)
  if origin is np.ndarray:
    dtype = typing.get_args(full_type)[1]
    np_type = typing.get_args(dtype)[0]
    return np.array([np_type()])
  if inspect.isclass(full_type) and issubclass(full_type, tf.Tensor):
    return tf.constant([1., 2.])

  if origin is None:  # _BasicPrimitiveType | None
    return full_type()  # type: ignore[call-arg,misc]

  # Types of the form Sequence[T]
  arguments = typing.get_args(full_type)
  if origin not in [list, tuple]:
    origin = list
  assert arguments
  return origin([_get_mock_value(arguments[0])] * len(arguments))


class Example(immutabledict.ImmutableOrderedDict[str, base_feature.Feature]):
  """Represention of an example."""

  def __init__(self, *_, **_2):
    for v in self.values():
      if isinstance(v, _CustomFeature):
        v.container = self
    self._validate()

  def _validate(self):
    """Raises a ValueError in any case a parameter is invalid."""
    for k, v in self.items():
      if not isinstance(k, str) or k == _INDEX:
        raise ValueError(f'Key {k} is invalid.')
      if _TF_EXAMPLE_SEPARATOR_CHAR in k:
        raise ValueError(
            f'Feature keys must avoid using {_TF_EXAMPLE_SEPARATOR_CHAR}".'
        )
      _FeatureWrapper(v).validate()  # Validate if the value is invalid.

  @property
  def num_custom_features(self) -> int:
    count = 0
    for feature in self.values():
      if isinstance(feature, _CustomFeature):
        count += 1
    return count

  def get_config(self) -> dict[str, _FeatureConfig]:
    """Returns the configuration of this Example.

    Returns:
      A mapping from user-provided key to the value's configuration.
    """
    # We should add an extra _INDEX key to the dictionary, so we can restore the
    # order of features. The value of this key is a mapping from each user key
    # to the index of it inside this OrderedDict.
    config: dict[str, _FeatureConfig] = {_INDEX: {}}
    for i, (k, v) in enumerate(self.items()):
      config[k] = _FeatureWrapper(v).get_config()
      config[_INDEX][k] = i
    return config

  @classmethod
  def from_config(cls, config: Mapping[str, _FeatureConfig]) -> Example:
    """Instantiates an Example from its config.

    Args:
      config: The configuration of the example.

    Returns:
      An Example instance whose config is identical to the given config.
    """
    features = {}
    for k, v in config.items():
      if k == _INDEX:
        continue
      features[k] = _FeatureWrapper.from_config(v).feature

    indices: dict[str, int] = config[_INDEX]  # type: ignore[assignment]
    return cls({
        k: features[k]
        for k in sorted(features.keys(), key=lambda name: indices[name])
    })

  def get_finalized_tf_example_io_spec(self) -> dict[str, Any]:
    """Converts this example to the spec of the finalized tf.Example.

    The output for this method is to be fed into tf.io.parse_example to allow
    parsing the finalized tf.Example as part of a TF graph.

    Returns:
      A mapping of the form {tf_example_key: tf.io.FixedLenFeature(), ...} that
      corresponds to keys and values of the finalized Example.
    """
    mock_example = {}
    for feature_key, feature in self.items():
      wrapped = _FeatureWrapper(feature)
      for output_key, output_full_type in wrapped.output_signature.items():
        if output_key:
          finalized_feature_key = nested_key(feature_key, output_key)
        else:
          finalized_feature_key = feature_key
        mock_example[finalized_feature_key] = _get_mock_value(output_full_type)

    mock_tf_example = Example(mock_example).to_tf_example()
    io_spec = {}
    for tf_key in mock_tf_example.features.feature:
      *_, primary_str, inner_str = tf_key.split(_TF_EXAMPLE_SEPARATOR_CHAR)
      primary_type = _TYPE_NAME_TO_TYPE[primary_str]
      inner_type = _TYPE_NAME_TO_TYPE[inner_str]
      io_spec[tf_key] = _types_to_io_spec(primary_type, inner_type)
    return io_spec

  def to_tf_example(self) -> tf.train.Example:
    """Converts this Example to a tf.train.Example.

    Returns:
      A tf.train.Example where each tf.train.Feature corresponds to a single
      value from a single feature configuration of self.get_config().
    """
    # This method converts self.get_config() to a tf.train.Example. These only
    # hold bytes, int64 and float lists. This means that we must save some
    # additional information for each tf.train.Feature to fully restore
    # self.get_config(). To that end, we'll save the primary and inner types
    # to each tf.train.Feature's name.
    tf_features = {}
    config = self.get_config()
    for config_key, feature_config in config.items():
      for feature_config_key, feature_config_value in feature_config.items():
        # These feature_config_values came from a _FeatureWrapper.get_config
        # so assume they're legit, especially since it might be big and take
        # precious time to validate.
        wrapped = _FeatureWrapper(feature_config_value)
        tf_key = _TF_EXAMPLE_SEPARATOR_CHAR.join([
            config_key,
            feature_config_key,
            wrapped.get_type(),
            wrapped.get_inner_type(),
        ])
        tf_features[tf_key] = wrapped.get_tf_feature()
    return tf.train.Example(features=tf.train.Features(feature=tf_features))

  @classmethod
  def from_tf_example(cls, tf_example: tf.train.Example) -> Example:
    """Instantiates an Example from a tf.train.Example.

    Args:
      tf_example: The tf.train.Example to use.

    Returns:
      An Example instance whose matching tf.train.Example is identical to the
      given tf_example.
    """
    config: Mapping[str, dict] = collections.defaultdict(dict)
    for key, tf_feature in tf_example.features.feature.items():
      config_key, feature_config_key, primary_type, inner_type = key.rsplit(
          _TF_EXAMPLE_SEPARATOR_CHAR, 3
      )
      config[config_key][feature_config_key] = _convert_tf_feature_by_types(
          tf_feature,
          primary_type,
          inner_type,
      )
    return cls.from_config(config)

  def to_bytes(self) -> bytes:
    """Returns an encoded representation for this Example."""
    return self.to_tf_example().SerializeToString()

  @classmethod
  def from_bytes(cls, serialized: bytes) -> Example:
    """Returns an Example from an encoded representation."""
    return cls.from_tf_example(tf.train.Example.FromString(serialized))

  def to_db_encoded(self) -> bytes:
    """Returns an encoded representation suitable for spanner."""
    return base64.b64encode(self.to_bytes())

  @classmethod
  def from_db_encoded(cls, encoded: bytes) -> Example:
    return cls.from_bytes(base64.b64decode(encoded))

  def finalize(self) -> Example:
    """Filters the given example according to output_signatures."""
    custom_features: list[str] = []
    output_keys: list[str] = []
    for feature_key, feature in self.items():
      if isinstance(feature, _CustomFeature):
        custom_features.append(feature_key)
        if not feature.drop_on_finalize:
          output_keys.extend(
              nested_key(feature_key, signature_key)
              for signature_key in feature.output_signature.keys()
          )
        # Nullify the value of feature.container since it is not needed anymore
        # after finalization. Without this, this example will always have a
        # reference to it, so GC will never collect it.
        if feature.container == self:
          feature.container = {}
      elif not any(
          feature_key.startswith(nested_key(k, '')) for k in custom_features
      ):
        output_keys.append(feature_key)
      continue
    return self.__class__({k: self[k] for k in output_keys})

  def __reduce__(self) -> tuple[Any, ...]:
    return (self.__class__, (self._dict,))

  def __str__(self) -> str:
    lines = ['Example:']
    for k, v in self.items():
      if not isinstance(v, _CustomFeature):
        lines.append(f'  {k}: {v}')
        continue
      lines.append(f'  {k}: {v.__class__.__name__}')
      for sub_k, sub_v in v._kwargs.items():
        if not isinstance(sub_v, serializable.Serializable):
          lines.append(f'    {sub_k}: {sub_v}')
          continue
        lines.append(f'    {sub_k}: {sub_v.__class__.__name__}')
        serialized = sub_v.serialize()
        try:
          text = serialized.decode()
        except UnicodeDecodeError:
          text = serialized.hex()
        lines.append(f'      {sub_k}.serialize(): {text}')
    return '\n'.join(_wrap_lines(lines))


def _wrap_lines(lines: Sequence[str]) -> list[str]:
  """Wraps all lines so they will not exceed _MAX_LINE_LENGTH in each line.

  Args:
    lines: The lines to wrap.

  Returns:
    A list of the wrapped lines.
  """
  new_lines = []
  for line in lines:
    if len(line) <= _MAX_LINE_LENGTH:
      new_lines.append(line)
      continue
    num_spaces = len(line) - len(line.lstrip(' '))
    text = line[num_spaces:]
    length_in_line = _MAX_LINE_LENGTH - num_spaces
    num_parts = int(np.ceil(len(text) / length_in_line))
    new_lines += [
        ' ' * num_spaces + text[i * length_in_line: (i + 1) * length_in_line]
        for i in range(num_parts)
    ]
  return new_lines


class _FeatureWrapper:
  """A wrapper that provides a unified API for both types of Features.

  Attributes:
    feature: The wrapped feature (either a ValueFeature or a CustomFeature.)
  """

  def __init__(self, feature: base_feature.Feature):
    self.feature = feature

  def validate(self):
    """Raises a ValueError in case self.feature is invalid."""
    feature = self.feature
    if not _generic_isinstance(feature, base_feature.Feature):
      raise ValueError(f'Value {feature} is not a valid feature.')
    if isinstance(feature, np.ndarray):
      dtype_name = feature.dtype.name
      if dtype_name not in _STRING_TO_NUMPY_TYPE and not (
          dtype_name.startswith('str') or dtype_name.startswith('bytes')
      ):
        raise ValueError(f'{feature.dtype} is not supported for np arrays.')
    if isinstance(feature, tf.Tensor):
      if feature.dtype not in [tf.float16, tf.float32, tf.float64]:
        raise ValueError(
            'Only float Tensors are supported. Use np arrays for other dtypes.'
        )
    if not isinstance(feature, (str, bytes)):
      if isinstance(feature, Sequence) and not feature:
        raise ValueError(
            'Empty sequences are not allowed, their type cannot be determined.'
        )
      if _generic_isinstance(
          feature, Sequence[base_feature.BasicCompositeType]
      ):
        # There's an issue with sequences of arrays / Tensors, since we can't
        # derive all the relevant tf.train.Example keys from a template example.
        # This is because each array in such a sequence corresponds to 2
        # additional keys in the tf.train.Example, so we must know the exact
        # number of elements in advance.
        # TODO(itayr): Remove this error the issue is resolved.
        raise ValueError(
            'Sequences of np.ndarrays / tf.Tensors are not yet supported.'
        )
      if (
          isinstance(feature, _CustomFeature) and
          any(_TF_EXAMPLE_SEPARATOR_CHAR in k for k in feature.output_signature)
      ):
        raise ValueError(
            'CustomFeatures cannot use {_TF_EXAMPLE_SEPARATOR_CHAR} in outputs.'
        )
    if self.get_type() not in _TYPE_NAME_TO_TYPE:
      raise ValueError(f'Unknown feature type {self.get_type()}.')


  def get_type(self) -> str:
    return type(self.feature).__name__

  def get_inner_type(self) -> str:
    if type(self.feature) in [bytes, str]:
      return self.get_type()
    if isinstance(self.feature, Sequence):
      return type(self.feature[0]).__name__
    return self.get_type()

  def _type_information(self) -> _FeatureConfig:
    return {
        _FEATURE_TYPE: self.get_type(),
        _FEATURE_SECONDARY_TYPE: self.get_inner_type(),
    }

  def get_config(self) -> _FeatureConfig:
    """Converts any type of Feature to its config.

    Returns:
      A _FeatureConfig matching self.feature.
    """
    value = self.feature
    config = self._type_information()
    if isinstance(value, _CustomFeature):
      return config | value.get_config()
    if _generic_isinstance(value, _PrimitiveType):
      return config | {_PLACEHOLDER: value}  # type: ignore[return-value]

    # The feature is a np.array or a tf.Tensor.
    if isinstance(value, tf.Tensor):
      np_array = value.numpy()
    else:
      np_array = value  # type: ignore[assignment]

    np_dtype = np_array.dtype.name
    serialized = tf.io.serialize_tensor(np_array).numpy()
    if np_dtype.startswith('bytes') or np_dtype.startswith('str'):
      config['serialized_bytes_0'] = zlib.compress(serialized)
    else:
      config['serialized_0'] = zlib.compress(serialized)
    config['dtype_0'] = np_dtype
    return config

  @classmethod
  def from_config(cls, config: _FeatureConfig) -> _FeatureWrapper:
    """Instantiates a _FeatureWrapper from its config.

    Args:
      config: The configuration of the _FeatureWrapper.

    Returns:
      A _FeatureWrapper whose config equals config.
    """
    config = config.copy()
    feature_type: str = config.pop(_FEATURE_TYPE)  # type: ignore[assignment]
    inner_type: str = config.pop(
        _FEATURE_SECONDARY_TYPE
    )  # type: ignore[assignment]

    if issubclass(_TYPE_NAME_TO_TYPE[feature_type], _CustomFeature):
      feature_class: type[_CustomFeature] = (
          _TYPE_NAME_TO_TYPE[feature_type]  # type: ignore[assignment]
      )
      return cls(feature_class.from_config(config))

    if _PLACEHOLDER in config:
      # This is a primitive type, since np.arrays / tf.Tensors have at least
      # serialized and dtype as keys in their config.
      return cls(config[_PLACEHOLDER])

    values = []
    for i in range(len(config) // 2):
      dtype_key = f'dtype_{i}'
      assert dtype_key in config
      dtype_name: str = config[dtype_key]  # type: ignore[assignment]
      if dtype_name.startswith('bytes') or dtype_name.startswith('str'):
        dtype: type = np.string_
        serialized_key = f'serialized_bytes_{i}'
      else:
        dtype = _STRING_TO_NUMPY_TYPE[dtype_name]  # type: ignore[index]
        serialized_key = f'serialized_{i}'
      assert serialized_key in config
      decompressed = zlib.decompress(
          config[serialized_key]  # type: ignore[arg-type]
      )
      values.append(
          tf.io.parse_tensor(decompressed, dtype).numpy().astype(dtype)
      )
    if inner_type == 'EagerTensor':
      values = [tf.convert_to_tensor(np_array) for np_array in values]
    return cls(
        _maybe_convert_to_scalar_or_tuple(values, feature_type, inner_type)
    )

  @property
  def output_signature(self) -> Mapping[str, type[_ValueFeature]]:
    """Extends base_feature.CustomFeature's output_signature.

    Returns:
      A mapping that states the keys and full types for the future finalized
      feature. In case the finalized feature only contains a single, not named
      subfeature, the key '' is used.
    """
    if isinstance(self.feature, _CustomFeature):
      if self.feature.drop_on_finalize:
        return {}
      return self.feature.output_signature

    # This is a ValueFeature, so its finalized version is just itself.
    if self.feature is None:
      return {'': type(None)}
    if isinstance(self.feature, typing.get_args(_BasicPrimitiveType)):
      return {'': type(self.feature)}
    if isinstance(self.feature, np.ndarray):
      # pylint: disable-next=unsubscriptable-object
      return {'': np.typing.NDArray[self.feature.dtype.type]}  # type: ignore
    if isinstance(self.feature, tf.Tensor):
      return {'': tf.Tensor}
    assert isinstance(self.feature, Sequence)
    sequence_type = type(self.feature)
    inner_value = more_itertools.first(self.feature)
    if sequence_type == tuple:
      return {'': tuple[type(inner_value), ...]}  # type: ignore[misc]
    return {'': sequence_type[type(inner_value)]}  # type: ignore[index]

  def get_tf_feature(self) -> tf.train.Feature:
    """Returns a tf.train.Feature which wraps this feature.

    This method assumes the _FeatureWrapper is valid.
    """
    values = self.feature
    if values is None:
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[]))

    if isinstance(values, typing.get_args(_BasicPrimitiveType)):
      values = [values]  # type: ignore[assignment]

    mock_value = values[0]
    if isinstance(mock_value, (int, bool)):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    if isinstance(mock_value, float):
      return tf.train.Feature(float_list=tf.train.FloatList(value=values))
    assert isinstance(mock_value, (str, bytes))
    if isinstance(mock_value, str):
      values = [v.encode() for v in values]  # type: ignore[union-attr]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def nested_key(feature_name: str, sub_key: str) -> str:
  return f'{feature_name}{_NESTING_CHAR}{sub_key}'


def _maybe_convert_to_scalar_or_tuple(
    values: Sequence[base_feature.BasicValue],
    primary_type: str,
    inner_type: str
) -> _ValueFeature:
  """Recovers the original value using the provided types."""
  if primary_type == 'tuple':
    return tuple(values)  # type: ignore[return-value]
  # Take care of the case where we have a value that's contained in a sequence
  # for no good reason (for example [tf.Tensor] when the primary_type is not a
  # sequence)
  if len(values) == 1 and primary_type == inner_type:
    return more_itertools.one(values)
  return values  # type: ignore[return-value]


def _convert_tf_feature_by_types(
    tf_feature: tf.train.Feature, primary_type: str, inner_type: str
) -> _ValueFeature:
  """Recovers the original value using the provided types."""
  if primary_type == 'NoneType':
    return None

  if tf_feature.bytes_list.value:
    values: list[_BasicPrimitiveType] = list(tf_feature.bytes_list.value)
  elif tf_feature.float_list.value:
    values = list(tf_feature.float_list.value)
  else:
    values = list(tf_feature.int64_list.value)

  if inner_type == 'str':
    values = [v.decode() for v in values]  # type: ignore[union-attr]
  elif inner_type == 'bool':
    values = [bool(v) for v in values]
  return _maybe_convert_to_scalar_or_tuple(values, primary_type, inner_type)


def _types_to_io_spec(feature_type: Any, inner_type: Any) -> Any:
  if feature_type == type(None):
    return tf.io.FixedLenFeature([0], tf.int64)
  # str and bytes are Sequences so deal with those first
  if feature_type in (str, bytes):
    return tf.io.FixedLenFeature([], tf.string)
  if issubclass(feature_type, Sequence):
    return tf.io.VarLenFeature(_PYTHON_TO_TF_FEATURE_TYPE[inner_type])
  return tf.io.FixedLenFeature([], _PYTHON_TO_TF_FEATURE_TYPE[feature_type])
