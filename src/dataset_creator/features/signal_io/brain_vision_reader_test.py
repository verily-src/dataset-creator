"""Tests for brain_vision_reader.py."""

import os

from absl.testing import parameterized  # type: ignore[import]

from dataset_creator.features.signal_io import brain_vision_reader

_THIS_DIR = os.path.dirname(__file__)
MULTIPLEXED_FILE = os.path.join(_THIS_DIR, 'testdata', 'test_VAmp.vhdr')
VECTORIZED_FILE = os.path.join(_THIS_DIR, 'testdata', 'brain_vision.vhdr')


class BrainVisionSignalReaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = brain_vision_reader.BrainVisionSignalReader(VECTORIZED_FILE)

  def test_num_leads(self):
    self.assertEqual(self.reader.num_leads, 64)

  def test_sampling_frequency(self):
    self.assertEqual(self.reader.sampling_frequency, 1000)

  def test_lead_headers_number_match_num_leads(self):
    self.assertLen(self.reader.lead_headers, self.reader.num_leads)

  def test_read_signal_returns_none_on_invalid_lead_num(self):
    self.assertIsNone(self.reader.read_signal(-1))

  def test_read_signal(self):
    signal = self.reader.read_signal(0)
    self.assertIsNotNone(signal)
    self.assertNotEmpty(signal)

  @parameterized.named_parameters(
      ('multiplexed', MULTIPLEXED_FILE), ('vectorized', VECTORIZED_FILE)
  )
  def test_read_signal_returns_right_signal_length(self, path: str):
    reader = brain_vision_reader.BrainVisionSignalReader(path)
    self.assertLen(reader.read_signal(0, 10, 15), 5)

  def test_total_samples(self):
    self.assertGreater(self.reader.total_samples, 0)
