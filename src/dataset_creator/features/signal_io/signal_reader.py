"""A module that exposes all SignalReaders publicly."""

from dataset_creator.features.signal_io import brain_vision_reader
from dataset_creator.features.signal_io import continuous_reader
from dataset_creator.features.signal_io import edf_reader
from dataset_creator.features.signal_io import eeglab_reader
from dataset_creator.features.signal_io import mne_reader
from dataset_creator.features.signal_io import simple_binary_reader
from dataset_creator.features.signal_io import wfdb_reader

EdfSignalReader = edf_reader.EdfSignalReader
BrainVisionSignalReader = brain_vision_reader.BrainVisionSignalReader
EEGLABSignalReader = eeglab_reader.EEGLABSignalReader
WfdbReader = wfdb_reader.WfdbReader
ContinuousSignalReader = continuous_reader.ContinuousSignalReader
SimpleBinarySignalReader = simple_binary_reader.SimpleBinarySignalReader
MneSignalReader = mne_reader.MneSignalReader
