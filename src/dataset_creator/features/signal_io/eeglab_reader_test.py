"""Tests for eeglab_reader.py."""

import os

from absl.testing import parameterized  # type: ignore[import]

from dataset_creator.features.signal_io import eeglab_reader

THIS_DIR = os.path.dirname(__file__)
V5_FILE = os.path.join(THIS_DIR, 'testdata', 'sub-S1_task-unnamed_eeg.set')
V5_ONEFILE = os.path.join(THIS_DIR, 'testdata', 'test_epochs_onefile.set')
V7_FILE = os.path.join(THIS_DIR, 'testdata', 'test_raw_onefile_h5.set')
V7_MULTITRIAL = os.path.join(THIS_DIR, 'testdata', 'test_epochs_onefile_h5.set')


class EEGLABSignalReaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = eeglab_reader.EEGLABSignalReader(V5_FILE)

  def test_num_leads(self):
    self.assertEqual(self.reader.num_leads, 24)

  def test_sampling_frequency(self):
    self.assertEqual(self.reader.sampling_frequency, 300)

  def test_lead_headers_number_match_num_leads(self):
    self.assertLen(self.reader.lead_headers, self.reader.num_leads)

  def test_read_signal_returns_none_on_invalid_lead_num(self):
    self.assertIsNone(self.reader.read_signal(-1))

  def test_read_signal(self):
    signal = self.reader.read_signal(0)
    self.assertIsNotNone(signal)
    self.assertNotEmpty(signal)

  @parameterized.named_parameters(
      ('v5', V5_FILE),
      ('v5_onefile', V5_ONEFILE),
      ('v7', V7_FILE),
      ('v7_multitrial', V7_MULTITRIAL),
  )
  def test_read_signal_returns_right_signal_length(self, path: str):
    reader = eeglab_reader.EEGLABSignalReader(path)
    self.assertLen(reader.read_signal(0, 1, 6), 5)

  def test_total_samples(self):
    self.assertGreater(self.reader.total_samples, 0)

