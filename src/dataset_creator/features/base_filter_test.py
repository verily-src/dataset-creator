"""Tests for base_filter.py."""

from typing import Any

from absl.testing import parameterized  # type: ignore[import]
import numpy as np

from dataset_creator.features import base_filter

# pylint: disable=protected-access

# Re-define some constants for simplicity of use
LOWPASS = base_filter.FilterType.LOW_PASS
BANDPASS = base_filter.FilterType.BAND_PASS
HIGHPASS = base_filter.FilterType.HIGH_PASS
BUTTER = base_filter.WindowType.BUTTER
CHEBY1 = base_filter.WindowType.CHEBY1
CHEBY2 = base_filter.WindowType.CHEBY2
ELLIP = base_filter.WindowType.ELLIP
BESSEL = base_filter.WindowType.BESSEL


class FilterConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "lowpass",
          f"{LOWPASS}|2|5|2.0|1.0|{BUTTER}|20",
          base_filter.FilterConfig(
              filter_type=LOWPASS,
              order=2,
              cutoff=5,
              sampling_frequency=20,
              window_type=BUTTER,
              ripple=2.0,
              min_attenuation=1.0,
          ),
      ),
      (
          "bandpass",
          f"{BANDPASS}|2|4,8|2.0|1.0|{CHEBY1}|20",
          base_filter.FilterConfig(
              filter_type=BANDPASS,
              order=2,
              cutoff=(4, 8),
              sampling_frequency=20,
              window_type=CHEBY1,
              ripple=2.0,
              min_attenuation=1.0,
          ),
      ),
      (
          "highpass",
          f"{HIGHPASS}|2|5|2.0|1.0|{CHEBY2}|20",
          base_filter.FilterConfig(
              filter_type=HIGHPASS,
              order=2,
              cutoff=5,
              sampling_frequency=20,
              window_type=CHEBY2,
              ripple=2.0,
              min_attenuation=1.0,
          ),
      ),
  )
  def test_filter_config_repr(
      self, expected_repr: str, config: base_filter.FilterConfig
  ):
    self.assertEqual(
        str(config),
        expected_repr,
    )

  @parameterized.named_parameters(
      (
          "lowpass_cutoff",
          {
              "filter_type": LOWPASS,
              "order": 2,
              "cutoff": (2, 4),
              "sampling_frequency": 20,
              "window_type": BUTTER,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          "cutoff must be scalar.",
      ),
      (
          "bandpass_cutoff",
          {
              "filter_type": BANDPASS,
              "order": 2,
              "cutoff": 5,
              "sampling_frequency": 20,
              "window_type": CHEBY2,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          "When using BandPass, cutoff must be a sequence of length 2",
      ),
      (
          "highpass_cutoff",
          {
              "filter_type": HIGHPASS,
              "order": 2,
              "cutoff": (2, 4),
              "sampling_frequency": 20,
              "window_type": BESSEL,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          "cutoff must be scalar.",
      ),
      (
          "sampling_frequency_bandpass_cutoff",
          {
              "filter_type": BANDPASS,
              "order": 2,
              "cutoff": (5, 10),
              "sampling_frequency": 10,
              "window_type": ELLIP,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          (
              "cutoff must smaller than the Nyquist frequency"
              " which is 5.0."
          ),
      ),
      (
          "sampling_frequency_bandpass_too_many_cutoffs",
          {
              "filter_type": BANDPASS,
              "order": 2,
              "cutoff": (5, 10, 15),
              "sampling_frequency": 10,
              "window_type": ELLIP,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          "When using BandPass, cutoff must be a sequence of length 2",
      ),
      (
          "sampling_frequency_cutoff_too_high",
          {
              "filter_type": LOWPASS,
              "order": 2,
              "cutoff": 8,
              "sampling_frequency": 10,
              "window_type": ELLIP,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          (
              "cutoff must smaller than the Nyquist frequency"
              " which is 5.0."
          ),
      ),
      (
          "sampling_frequency_bandpass_negative_cutoffs",
          {
              "filter_type": BANDPASS,
              "order": 2,
              "cutoff": (-5, 1),
              "sampling_frequency": 10,
              "window_type": ELLIP,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          "cutoff must be >= 0.",
      ),
      (
          "sampling_frequency_negative_cutoffs",
          {
              "filter_type": LOWPASS,
              "order": 2,
              "cutoff": -3,
              "sampling_frequency": 10,
              "window_type": BUTTER,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          "cutoff must be >= 0.",
      ),
      (
          "negative_sampling_frequency",
          {
              "filter_type": LOWPASS,
              "order": 2,
              "cutoff": 3,
              "sampling_frequency": -10,
              "window_type": BUTTER,
              "ripple": 1.0,
              "min_attenuation": 1.0,
          },
          ValueError,
          "sampling_frequency must be > 0.",
      ),
  )
  def test_config_post_init_validation(
      self,
      config_params: dict[str, Any],
      error_type: Exception,
      message: str,
  ):
    with self.assertRaisesRegex(error_type, message):
      base_filter.FilterConfig(**config_params)  # type: ignore[arg-type]


class BaseFilterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "lowpass",
          base_filter.FilterConfig(
              filter_type=LOWPASS,
              order=5,
              cutoff=15,
              sampling_frequency=600,
              window_type=CHEBY1,
              ripple=4.0,
              min_attenuation=4.0,
          ),
          0,
      ),
      (
          "bandpass",
          base_filter.FilterConfig(
              filter_type=BANDPASS,
              order=9,
              cutoff=(60, 140),
              sampling_frequency=600,
              window_type=CHEBY2,
              ripple=8.0,
              min_attenuation=2.0,
          ),
          1,
      ),
      (
          "highpass",
          base_filter.FilterConfig(
              filter_type=HIGHPASS,
              order=5,
              cutoff=160,
              sampling_frequency=600,
              window_type=CHEBY2,
              ripple=18.0,
              min_attenuation=1.0,
          ),
          2,
      ),
  )
  def test_filter_process(
      self, filter_config: base_filter.FilterConfig, base_component_index: int
  ):
    # Generate a toy signal
    low_frequency = 5  # [Hz]
    mid_frequency = 100  # [Hz]
    high_frequency = 200  # [Hz]
    sampling_frequency = 3 * high_frequency
    t = 2  # [sec]
    total_length = t * sampling_frequency
    time = np.arange(0, total_length) / sampling_frequency
    low_pass_component = (1 / 3) * np.sin(low_frequency * time)
    mid_pass_component = (1 / 3) * np.sin(mid_frequency * time)
    high_pass_component = (1 / 3) * np.sin(high_frequency * time)
    base_components = [
        low_pass_component,
        mid_pass_component,
        high_pass_component,
    ]
    signal = sum(base_components)

    # Spectral error threshold
    threshold = 0.8

    # Instantiate the filter
    test_filter = base_filter.BaseFilter(filter_config)

    # Filter the signal
    filtered_signal = test_filter.process(signal)

    # Check that the differences in the spectrum are small enough
    base_component = base_components[base_component_index]
    filtered_signal_fft = np.abs(np.fft.fft(filtered_signal))
    component_fft = np.abs(np.fft.fft(base_component))
    # TODO(yelul): Design better parameters for the BAND/HIGH PASS filters so we
    # can reduce the threshold to ~0.1
    diff = (
        np.abs(filtered_signal_fft - component_fft).sum() / component_fft.sum()
    )
    self.assertLessEqual(diff, threshold)

  @parameterized.named_parameters(
      (
          "lowpass",
          base_filter.FilterConfig(
              filter_type=LOWPASS,
              order=5,
              cutoff=15,
              sampling_frequency=600,
              window_type=CHEBY1,
              ripple=4.0,
              min_attenuation=4.0,
          ),
      ),
      (
          "bandpass",
          base_filter.FilterConfig(
              filter_type=BANDPASS,
              order=9,
              cutoff=(60, 140),
              sampling_frequency=600,
              window_type=CHEBY2,
              ripple=8.0,
              min_attenuation=2.0,
          ),
      ),
      (
          "highpass",
          base_filter.FilterConfig(
              filter_type=HIGHPASS,
              order=5,
              cutoff=160,
              sampling_frequency=600,
              window_type=CHEBY2,
              ripple=18.0,
              min_attenuation=1.0,
          ),
      ),
  )
  def test_serielization(self, filter_config: base_filter.FilterConfig):
    # Generate a filter
    test_filter = base_filter.BaseFilter(filter_config)

    # Serialize the filter
    serialized_filter = test_filter.serialize()

    # Deserialize the filter
    deserialized_filter = base_filter.BaseFilter.deserialize(serialized_filter)

    # Validate that the deserialized filter is identical to the original one
    config = test_filter._config
    deserialized_config = deserialized_filter._config

    self.assertEqual(
        config.filter_type.value, deserialized_config.filter_type.value
    )
    self.assertEqual(
        config.window_type.value, deserialized_config.window_type.value
    )
    self.assertEqual(config.order, deserialized_config.order)
    self.assertEqual(config.ripple, deserialized_config.ripple)
    self.assertEqual(
        config.min_attenuation, deserialized_config.min_attenuation
    )
    self.assertEqual(
        config.sampling_frequency,
        deserialized_config.sampling_frequency,
    )
    self.assertEqual(
        config.cutoff,
        deserialized_config.cutoff,
    )

# pylint: enable=protected-access
