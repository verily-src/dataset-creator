"""Tests for generated_dataset.py."""

import base64
import datetime

from absl.testing import parameterized  # type: ignore[import]
from google.cloud import spanner  # type: ignore[attr-defined]

from dataset_creator import example_lib
from dataset_creator import generated_dataset
from dataset_creator import helpers
from dataset_creator import test_utils
from dataset_creator.features import images_feature
from dataset_creator.features import inference_feature

# pylint: disable=protected-access

Example = example_lib.Example
GeneratedDataset = generated_dataset.GeneratedDataset


class GeneratedDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    db = helpers.get_db()
    # Since tearDown deletes the entire DB, let's make absolutely sure that
    # it is indeed the test DB.
    assert 'test-project-id' in db.name
    self.db = db
    self.timestamps_sequences = test_utils.default_video_timestamps_sequences()
    self.video_path = test_utils.mock_video_path()
    # Set tzinfo since inserting to spanner sets it if it's None
    now = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)
    keras_model_path, outputs_layer_names, example_to_inputs = (
        test_utils.inference_parameters()
    )

    self.examples = list(
        test_utils.get_examples_generator(
            video_timestamps_sequences=self.timestamps_sequences,
            video_path=self.video_path,
            keras_model_path=keras_model_path,
            outputs_layer_names=outputs_layer_names,
            example_to_inputs=example_to_inputs,
        ))

    self.generated_dataset = GeneratedDataset(
        dataset_name='dataset_creator_test',
        creation_time=now,
        dataset_creator_version='test',
        get_examples_generator=lambda: self.examples,
    )
    self.generated_dataset.save()

  def tearDown(self):
    with self.db.batch() as batch:
      batch.delete('GeneratedDatasets', keyset=spanner.KeySet(all_=True))

  def test_save_dataset_parameters(self):
    # Override the saved dataset with / without example bank
    self.generated_dataset.save()
    table_name = 'GeneratedDatasets'
    columns = ('DatasetName', 'CreationTime', 'DatasetCreatorVersion')

    with self.db.snapshot() as snapshot:
      keyset = spanner.KeySet(keys=[self.generated_dataset.key])
      row = snapshot.read(
        table=table_name, columns=columns, keyset=keyset
      ).one()
    self.assertEqual(row[0], self.generated_dataset.dataset_name)
    self.assertEqual(row[1], self.generated_dataset.creation_time)
    self.assertEqual(row[2], self.generated_dataset.dataset_creator_version)

  def test_get_creation_times(self):
    registered_datasets = generated_dataset.get_registered_datasets(
        self.generated_dataset.dataset_name)
    self.assertLen(registered_datasets, 1)
    self.assertEqual(registered_datasets[0][0],
                     self.generated_dataset.dataset_name)
    self.assertEqual(registered_datasets[0][1],
                     self.generated_dataset.creation_time)

  # Check that the last transaction has been written.
  def test_save_examples(self):
    columns = ('DatasetName', 'CreationTime', 'ExampleIndex', 'EncodedExample')

    example_index = len(self.timestamps_sequences) - 1
    with self.db.snapshot() as snapshot:
      keyset = spanner.KeySet(
          keys=[(*self.generated_dataset.key, example_index)]
      )
      row = snapshot.read(table=generated_dataset.EXAMPLES_TABLE,
                              columns=columns, keyset=keyset).one()
    self.assertEqual(row[0], self.generated_dataset.dataset_name)
    self.assertEqual(row[1], self.generated_dataset.creation_time)
    self.assertEqual(row[2], example_index)

    example = self.examples[example_index]
    read_example = Example.from_bytes(base64.b64decode(row[3]))
    self.assertEqualExample(example, read_example)

  def assertEqualExample(self, a: Example, b: Example):
    # Verify order and length of keys
    self.assertEqual(list(a.keys()), list(b.keys()))
    for value_a, value_b in zip(a.values(), b.values()):
      if isinstance(value_a, images_feature.ImagesFeature):
        self.assertEqual(
            value_a.get_config(),  # type: ignore[union-attr]
            value_b.get_config(),  # type: ignore[union-attr]
        )
      elif isinstance(value_a, inference_feature.InferenceFeature):
        # We can't simply check the config in this case, since this type of
        # CustomFeature contains a marshaled function, whose serialization is
        # not entirely deterministic.
        self.assertEqual(
            value_a._keras_model_path,  # type: ignore[union-attr]
            value_b._keras_model_path,  # type: ignore[union-attr]
        )
        self.assertCountEqual(
            value_a._outputs_layer_names,  # type: ignore[union-attr]
            value_b._outputs_layer_names,  # type: ignore[union-attr]
        )
        self.assertEqual(
            value_a._container_to_inputs(a),  # type: ignore[union-attr]
            value_b._container_to_inputs(b),  # type: ignore[union-attr]
        )
      else:
        self.assertEqual(value_a, value_b)

  def test_from_key(self):
    dataset = GeneratedDataset.from_key(self.generated_dataset.dataset_name,
                                        self.generated_dataset.creation_time)
    assert dataset is not None
    examples = list(dataset.get_examples_generator())
    # Expect 5 results since one timestamps_sequence corresponds to an image.
    self.assertLen(examples, 5)
    # Check that rows return in the right order
    self.assertEqual(
        [example[test_utils.LABELS_FEATURE_NAME] for example in examples],
        [example[test_utils.LABELS_FEATURE_NAME] for example in self.examples]
    )

  @parameterized.named_parameters(
      ('all_examples',
       0, 5,
       list(test_utils.default_video_timestamps_sequences())
       + [test_utils.timestamps_sequence_for_image()]),
      ('1_to_3', 1, 3, test_utils.default_video_timestamps_sequences()[1:3]),
      ('end_very_high',
       1, 10,
       list(test_utils.default_video_timestamps_sequences()[1:10])
       + [test_utils.timestamps_sequence_for_image()]),
      ('out_of_range', 10, 20, []),
  )
  def test_example_reader(self, from_index, until_index, expected_timestamps):
    examples = generated_dataset.ExampleReader(
        dataset_name=self.generated_dataset.dataset_name,
        creation_time=self.generated_dataset.creation_time,
        from_index=from_index,
        until_index=until_index).read()
    timestamps = [
        example[test_utils.IMAGES_FEATURE_NAME]._read_at
        for example in examples
    ]
    self.assertEqual(timestamps, expected_timestamps)

  def test_from_key_returns_none_invalid_key(self):
    self.assertIsNone(
        GeneratedDataset.from_key(
            'invalid_dataset_name', datetime.datetime.now()))

  def test_save_a_very_long_example(self):
    very_long_example = Example(
        video_path='.',
        timestamps_sequence=list(range(0, 10000))
    )
    now = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)
    GeneratedDataset(
        dataset_name='test',
        creation_time=now,
        dataset_creator_version='testversion',
        get_examples_generator=lambda: [very_long_example],
    ).save()
    self.assertIsNotNone(GeneratedDataset.from_key('test', now))

  def test_num_examples(self):
    self.assertLen(self.examples, self.generated_dataset.num_examples)

  def test_num_examples_returns_0_for_new_dataset(self):
    new_dataset = GeneratedDataset(
        dataset_name='test',
        creation_time=datetime.datetime.now(),
        dataset_creator_version='test',
        get_examples_generator=lambda: self.examples,
    )
    self.assertEqual(new_dataset.num_examples, 0)

  def test_from_key_populates_none_version(self):
    to_be_saved_with_none_version = GeneratedDataset(
        dataset_name='test',
        creation_time=datetime.datetime.now(),
        dataset_creator_version=None,
        get_examples_generator=lambda: self.examples,
    )
    to_be_saved_with_none_version.save()
    saved_dataset = GeneratedDataset.from_key(
        to_be_saved_with_none_version.dataset_name,
        to_be_saved_with_none_version.creation_time,
    )
    assert saved_dataset is not None
    self.assertEqual(saved_dataset.dataset_creator_version, '0.0.1')

# pylint: enable=protected-access
