"""Tests for continuous_reader.py."""

import os

from absl.testing import absltest  # type: ignore[import]

from dataset_creator.features.signal_io import continuous_reader

_THIS_DIR = os.path.dirname(__file__)


class ContinuousSignalReaderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = continuous_reader.ContinuousSignalReader(
        os.path.join(_THIS_DIR, 'testdata', 'scan41_short.cnt')
    )

  def test_num_leads(self):
    self.assertEqual(self.reader.num_leads, 128)

  def test_sampling_frequency(self):
    self.assertEqual(self.reader.sampling_frequency, 400)

  def test_lead_headers_number_match_num_leads(self):
    self.assertLen(self.reader.lead_headers, self.reader.num_leads)

  def test_read_signal(self):
    signal = self.reader.read_signal(0)
    self.assertIsNotNone(signal)
    self.assertNotEmpty(signal)

  def test_read_signal_returns_right_signal_length(self):
    self.assertLen(self.reader.read_signal(0, 10, 15), 5)

  def test_total_samples(self):
    self.assertGreater(self.reader.total_samples, 0)
