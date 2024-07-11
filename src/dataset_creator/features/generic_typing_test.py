"""Tests for generic_typing."""

from typing import Any, Sequence, Union

from absl.testing import parameterized  # type: ignore[import]
import numpy as np
import numpy.typing  # pylint: disable=unused-import

from dataset_creator.features import generic_typing


class GenericTypingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', 1, int),
      ('list', [1, 'a'], list),
      ('list_of_ints', [1, 2, 3], list[int]),
      ('tuple_of_bools', (True, False), tuple[bool, bool]),
      ('general_sequence', (1, '23'), Sequence),
      ('union', 4, Union[int, str]),
      ('None', None, type(None)),
      ('sequence_of_union', [1, '23'], Sequence[Union[int, str]]),
      ('mixed_tuple', (1, 'a'), tuple[int, str]),
      ('with_ellipsis', (1, 2, 3), tuple[int, ...]),
      ('nested_types', ((1, 2), ('3',)), tuple[tuple[int, int], tuple[str]]),
      ('np_array', np.array([1], dtype=np.int16), np.typing.NDArray[np.int16]),
      ('non_typed_np_array', np.array([1], dtype=np.int32), np.ndarray)
  )
  def test_matching_types(self, value, typing_type):
    self.assertTrue(generic_typing.generic_isinstance(value, typing_type))

  @parameterized.named_parameters(
      ('str_with_int', '1', int),
      ('bad_arg', (1, 2), tuple[str]),
      ('list_with_tuple', [1, 2], tuple[int]),
      ('not_included_in_union', '123', Union[int, bytes]),
      ('tuple_length_mismatch', (1, 2, 3), tuple[int, int]),
      ('tuple_with_bad_types_order', (1, 'a'), tuple[str, int]),
      ('nested_type_doesnt_match',
       ((1,), ('3',)), tuple[tuple[int], tuple[float]]),
      ('np_array', np.array([], dtype=np.int16), np.typing.NDArray[np.int32]),
      ('non_np_array', [1, 2, 3], np.typing.NDArray[np.int32]),
  )
  def test_non_matching_types(self, value, typing_type):
    self.assertFalse(generic_typing.generic_isinstance(value, typing_type))

  @parameterized.named_parameters(
      (
          'invalid_ellipsis',
          (0,),
          tuple[int, float, ...],  # type: ignore[misc]
          'ellipsis annotation'
      ),
      ('not_supported_type', {'1': '2'}, dict[str, str], 'is not supported'),
  )
  def test_invalid_type_raises_valueerror(
      self, value: Any, tp: type[Any], error_msg: str
  ):
    with self.assertRaisesRegex(ValueError, error_msg):
      generic_typing.generic_isinstance(value, tp)
