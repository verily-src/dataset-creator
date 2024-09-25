"""Tests for gcp_database.py."""

import datetime

from absl.testing import absltest  # type: ignore[import]
import apache_beam as beam
from apache_beam.runners.interactive import interactive_beam
from apache_beam.runners.interactive import interactive_runner
from google.cloud import spanner  # type: ignore[attr-defined]

from dataset_creator import example_lib
from dataset_creator.environments.gcp import gcp_database

# pylint: disable=protected-access

DATASET_CREATION_TIME = datetime.datetime.now(tz=datetime.timezone.utc)
DATASET_NAME = 'test_dataset'


def setUpModule():
  spanner_db = gcp_database.SpannerDatabase()
  assert 'test-project-id' in spanner_db._db.name
  for prefix in ('', 'another_'):
    dataset_name = f'{prefix}{DATASET_NAME}'
    spanner_db.insert(
        table_name='GeneratedDatasets',
        columns=('DatasetName', 'CreationTime', 'DatasetCreatorVersion'),
        rows=[(dataset_name, DATASET_CREATION_TIME, 'test')],
    )
    examples = [example_lib.Example({'label': i}) for i in range(3)]
    spanner_db.insert(
        table_name='DatasetUnpopulatedExamples',
        columns=(
            'DatasetName', 'CreationTime', 'ExampleIndex', 'EncodedExample'
        ),
        rows=[
            (dataset_name, DATASET_CREATION_TIME, i, example.to_db_encoded())
            for i, example in enumerate(examples)
        ],
    )


def tearDownModule():
  spanner_db = gcp_database.SpannerDatabase()
  assert 'test-project-id' in spanner_db._db.name
  with spanner_db._db.batch() as batch:
    batch.delete('GeneratedDatasets', keyset=spanner.KeySet(all_=True))
    batch.delete('DatasetUnpopulatedExamples', keyset=spanner.KeySet(all_=True))


class SpannerDatabaseTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.spanner_db = gcp_database.SpannerDatabase()

  def test_read_with_key_range(self):
    examples = self.spanner_db.read(
        table_name='DatasetUnpopulatedExamples',
        columns=('EncodedExample',),
        range_key_start=(DATASET_NAME, DATASET_CREATION_TIME, 0),
        range_key_end=(DATASET_NAME, DATASET_CREATION_TIME, 2),
    )
    # range_key_end is exclusive, so we should get 2 examples.
    self.assertLen(list(examples), 2)

  def test_read_with_filter_key(self):
    examples = self.spanner_db.read(
        table_name='GeneratedDatasets',
        columns=('DatasetName',),
        filter_key=(DATASET_NAME, DATASET_CREATION_TIME),
    )
    self.assertLen(list(examples), 1)

  def test_query(self):
    all_examples = self.spanner_db.query(
        'SELECT * FROM DatasetUnpopulatedExamples',
    )
    # 3 examples per dataset saved in the DB.
    self.assertLen(list(all_examples), 6)

  def test_create_read_rows_ptransform(self):
    read_ptransform = self.spanner_db.create_read_rows_ptransform(
        range_key_start=(DATASET_NAME, DATASET_CREATION_TIME, 0),
        range_key_end=(DATASET_NAME, DATASET_CREATION_TIME, 3),
    )
    pipeline = beam.Pipeline(interactive_runner.InteractiveRunner())
    output_pcollection = pipeline | 'Read' >> read_ptransform
    interactive_beam.watch(locals())
    result = pipeline.run()
    result.wait_until_finish()
    self.assertLen(result.get(output_pcollection), 3)

# pylint: enable=protected-access
