"""Tests for multi_lead_signal_feature.py."""

import os
from typing import Optional, Sequence

from absl.testing import parameterized  # type: ignore[import]
import numpy as np
import tensorflow as tf

from dataset_creator.features import base_filter
from dataset_creator.features import fields
from dataset_creator.features import generic_typing
from dataset_creator.features import multi_lead_signal_feature
from dataset_creator.features.signal_io import base_signal_reader
from dataset_creator.features.signal_io import signal_reader

_THIS_DIR = os.path.dirname(__file__)

_MultiLeadSignalFeature = multi_lead_signal_feature.MultiLeadSignalFeature


class MultiLeadSignalFeatureTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.reader = signal_reader.EdfSignalReader(
        os.path.join(
            _THIS_DIR,
            'signal_io/testdata',
            'test_edf_overlapping_annotations.edf'
        )
    )
    self.feature = _MultiLeadSignalFeature(
        self.reader, resample_at=10, start=0, end=250
    )
    self.mock_signals = [
        np.zeros(10, dtype=np.float32) for _ in range(self.reader.num_leads)
    ]

  def test_split(self):
    self.assertGreater(len(self.feature.split()), 1)

  def test_process(self):
    signal = self.feature.process(0, None)
    self.assertIsNotNone(signal)
    self.assertGreater(signal.size, 0)

  def test_process_returns_none_with_invalid_channel_number(self):
    self.assertIsNone(self.feature.process(1000, None))

  def test_merge(self):
    split = self.feature.split()
    signals = [self.feature.process(i, None) for i in split]
    merged = self.feature.merge(signals)
    self.assertLen(merged[fields.MULTI_LEAD_LABELS], len(split))
    self.assertLen(merged[fields.MULTI_LEAD_SIGNAL], len(split))
    self.assertEqual(merged[fields.MULTI_LEAD_SIGNAL].dtype, np.float32)

  def test_merge_with_no_signals(self):
    self.assertEmpty(self.feature.merge([]))

  def test_output_signature_keys_are_a_subset_of_merge_output_keys(self):
    merged_keys = self.feature.merge(self.mock_signals).keys()
    self.assertContainsSubset(self.feature.output_signature.keys(), merged_keys)

  def test_output_signature_types_match_merged_types(self):
    merged = self.feature.merge(self.mock_signals)
    output_signature = self.feature.output_signature
    matches = [
        generic_typing.generic_isinstance(merged[k], v)
        for k, v in output_signature.items()
    ]
    self.assertTrue(all(matches))

  @parameterized.named_parameters(
      ('empty_matches', ['this_channel_doesnt_exist']),
      ('too_many_matches', ['.*']),
  )
  def test_split_returns_an_empty_sequence(self, patterns: Sequence[str]):
    feature = _MultiLeadSignalFeature(self.reader, label_patterns=patterns)
    self.assertEmpty(feature.split())

  def test_split_returns_a_channel_when_including_empty_leads(self):
    feature = _MultiLeadSignalFeature(
        self.reader,
        label_patterns=['this_channel_doesnt_exist'],
        empty_leads='empty'
    )
    self.assertLen(feature.split(), 1)

  @parameterized.named_parameters(
      ('no_empty_leads', None),
      ('empty_leads', 'empty')
  )
  def test_processing_of_reference_lead(self, empty_leads: Optional[str]):
    feature = _MultiLeadSignalFeature(
        self.reader,
        label_patterns=['Fp1', 'TestReference'],
        reference_lead='TestReference',
        empty_leads=empty_leads
    )
    # pylint: disable-next=unbalanced-tuple-unpacking
    fp1_entry, reference_entry = feature.split()
    ref_signal = feature.process(reference_entry, None)
    self.assertIsNotNone(ref_signal)
    self.assertAllClose(ref_signal, np.zeros_like(ref_signal))
    fp1_signal = feature.process(fp1_entry, None)
    assert fp1_signal is not None and ref_signal is not None
    self.assertNotEmpty(feature.merge([fp1_signal, ref_signal]))

  @parameterized.named_parameters(
      ('no_empty_leads', None),
      ('empty_leads', 'empty')
  )
  def test_merged_length_matches_patterns_length(self, empty_leads: str):
    feature = _MultiLeadSignalFeature(
        self.reader, label_patterns=['Fp1'], empty_leads=empty_leads
    )
    merged = feature.merge(self.mock_signals)
    self.assertLen(merged[fields.MULTI_LEAD_LABELS], 1)

  @parameterized.named_parameters(
      ('no_empty_leads', None),
      ('empty_leads', 'empty')
  )
  def test_merged_matches_patterns_order(self, empty_leads: str):
    merged_without_patterns = self.feature.merge(self.mock_signals)
    no_pattern_labels = merged_without_patterns[fields.MULTI_LEAD_LABELS]
    feature = _MultiLeadSignalFeature(
        self.reader, label_patterns=['Fp2', 'Fp1'], empty_leads=empty_leads
    )
    labels: list[str] = feature.merge(
        self.mock_signals
    )[fields.MULTI_LEAD_LABELS]  # type: ignore[assignment]

    no_pattern_fp1_index = no_pattern_labels.index('Fp1.')
    no_pattern_fp2_index = no_pattern_labels.index('Fp2.')
    fp1_index = labels.index('Fp1.')
    fp2_index = labels.index('Fp2.')

    self.assertGreater(no_pattern_fp2_index, no_pattern_fp1_index)
    self.assertGreater(fp1_index, fp2_index)

  @parameterized.named_parameters(
      ('no_empty_leads', None),
      ('empty_leads', 'empty')
  )
  def test_signal_with_filter(self, empty_leads: str):
    config = base_filter.FilterConfig(
        filter_type=base_filter.FilterType.LOW_PASS,
        sampling_frequency=self.reader.sampling_frequency,
        cutoff=self.reader.sampling_frequency / 3,
    )
    feature = _MultiLeadSignalFeature(
        self.reader,
        signal_filter=base_filter.BaseFilter(config),
        empty_leads=empty_leads
    )
    self.assertNotAllClose(
        feature.process(0, None), self.feature.process(0, None)
    )

  @parameterized.named_parameters(
      ('edf_file', 'test.edf', signal_reader.EdfSignalReader),
      ('wfdb_file', 'test.hea', signal_reader.WfdbReader),
      ('eeglab_file', 'test.set', signal_reader.EEGLABSignalReader),
      ('brain_vision_file', 'test.vhdr', signal_reader.BrainVisionSignalReader),
      ('continuous_file', 'test.cnt', signal_reader.ContinuousSignalReader),
      ('raw_file', 'test.raw', signal_reader.SimpleBinarySignalReader),
      ('mff_dir', 'test.mff/', signal_reader.MneSignalReader),
  )
  def test_get_default_reader(
      self,
      file_path: str,
      expected_class: type[base_signal_reader.AbstractMultiLeadSignalReader]
  ):
    # Pass check_path=False since the files don't actually exist.
    reader = multi_lead_signal_feature.get_default_reader(
        file_path, check_path=False
    )
    self.assertIsInstance(reader, expected_class)

  def test_get_default_reader_with_bad_file_extension(self):
    with self.assertRaisesRegex(NotImplementedError, 'Unsupported extension'):
      multi_lead_signal_feature.get_default_reader('test.mp4', check_path=False)

  @parameterized.named_parameters(
      ('no_empty_leads', None),
      ('empty_leads', 'empty')
  )
  def test_split_returns_empty_sequence_with_invalid_file(
      self, empty_leads: str
  ):
    reader = signal_reader.EdfSignalReader('test.edf', check_path=False)
    feature = _MultiLeadSignalFeature(
        reader, resample_at=10, start=0, end=250, empty_leads=empty_leads
    )
    self.assertEmpty(feature.split())

  def test_empty_leads_values(self):
    config = base_filter.FilterConfig(
        filter_type=base_filter.FilterType.LOW_PASS,
        sampling_frequency=self.reader.sampling_frequency,
        cutoff=self.reader.sampling_frequency / 3,
    )
    feature = _MultiLeadSignalFeature(
        self.reader,
        signal_filter=base_filter.BaseFilter(config),
        label_patterns=['Fp2'] + ['impossible_lead_name'] * 100 + ['Fp1'],
        empty_leads='empty'
    )
    split = feature.split()
    signals = [feature.process(i, None) for i in split]
    merged = feature.merge(signals)
    average_values = merged[fields.MULTI_LEAD_SIGNAL][[0, -1]].mean(axis=0)

    self.assertTrue(np.all(merged[fields.MULTI_LEAD_SIGNAL][1:-1] == -1))
    self.assertAllClose(
        average_values, np.zeros_like(average_values), atol=1e-4,
    )
    self.assertLen(merged[fields.MULTI_LEAD_LABELS], len(split))
    self.assertLen(merged[fields.MULTI_LEAD_SIGNAL], len(split))
    self.assertEqual(merged[fields.MULTI_LEAD_SIGNAL].dtype, np.float32)
