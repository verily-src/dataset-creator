"""Tests for edf_reader.py."""

import os

from absl.testing import parameterized  # type: ignore[import]

from dataset_creator.features.signal_io import edf_reader

_THIS_DIR = os.path.dirname(__file__)
EDF_FILE = os.path.join(
    _THIS_DIR, 'testdata', 'test_edf_overlapping_annotations.edf'
)
BDF_FILE = os.path.join(_THIS_DIR, 'testdata', 'test_bdf_stim_channel.bdf')


class EdfReaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = edf_reader.EdfSignalReader(EDF_FILE)

  def test_instantiation_raises_filenotfounderror_with_invalid_path(self):
    with self.assertRaises(FileNotFoundError):
      _ = edf_reader.EdfSignalReader('/tmp/signal.edf', check_path=True)

  def test_instantiation_raises_valueerror_with_invalid_extension(self):
    with self.assertRaisesRegex(ValueError, 'Invalid file format'):
      _ = edf_reader.EdfSignalReader('/test/signal.xyz', check_path=False)

  def test_num_leads(self):
    self.assertEqual(self.reader.num_leads, 65)

  def test_sampling_frequency(self):
    self.assertEqual(self.reader.sampling_frequency, 128)

  def test_lead_headers_number_match_num_leads(self):
    self.assertLen(self.reader.lead_headers, self.reader.num_leads)

  def test_read_signal_returns_none_on_invalid_lead_num(self):
    self.assertIsNone(self.reader.read_signal(-1))

  def test_read_signal(self):
    signal = self.reader.read_signal(0)
    self.assertIsNotNone(signal)
    self.assertNotEmpty(signal)

  @parameterized.named_parameters(('edf', EDF_FILE), ('bdf', BDF_FILE))
  def test_read_signal_returns_right_signal_length(self, path: str):
    reader = edf_reader.EdfSignalReader(path)
    self.assertLen(reader.read_signal(0, 10, 15), 5)

  # ======== Tests for functionality in AbstractMultiLeadSignalReader ========:
  @parameterized.named_parameters(
      ('invalid_start', -1, 1),
      ('start_not_less_than_end', 0, 0),
  )
  def test_read_signal_with_invalid_params_returns_none(
      self, start: int, end: int
  ):
    self.assertIsNone(self.reader.read_signal(0, start, end))

  def test_total_samples(self):
    self.assertGreater(self.reader.total_samples, 0)

  def test_duration(self):
    self.assertGreater(self.reader.duration.total_seconds(), 0)

  def test_deserialization_returns_an_equivalent_reader(self):
    serialized = self.reader.serialize()
    deserialized_reader = edf_reader.EdfSignalReader.deserialize(serialized)
    self.assertEqual(deserialized_reader.path, self.reader.path)
    # pylint: disable-next=protected-access
    self.assertEqual(deserialized_reader._check_path, self.reader._check_path)
