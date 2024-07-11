"""Tests for simple_binary_reader.py."""

import os

from absl.testing import absltest  # type: ignore[import]

from dataset_creator.features.signal_io import mne_reader

_THIS_DIR = os.path.dirname(__file__)


class MneSignalReaderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = mne_reader.MneSignalReader(
        os.path.join(_THIS_DIR, 'testdata', 'test_egi.mff')
    )

  def test_num_leads(self):
    self.assertEqual(self.reader.num_leads, 136)

  def test_sampling_frequency(self):
    self.assertEqual(self.reader.sampling_frequency, 1000)

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
