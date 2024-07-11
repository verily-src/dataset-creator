"""A GeneratedDataset is a dataset that is already / is about to be saved."""

from __future__ import annotations

import dataclasses
import datetime
from typing import Iterable, Optional, Union

from google.cloud import spanner  # type: ignore[attr-defined]
from google.cloud.exceptions import exceptions  # type: ignore[import]
import more_itertools
import tqdm

from dataset_creator import example_lib
from dataset_creator import helpers

Example = example_lib.Example
RowNotFoundError = exceptions.NotFound

EXAMPLES_TABLE = 'DatasetUnpopulatedExamples'


@dataclasses.dataclass(frozen=True)
class GeneratedDataset:
  """Represents a dataset that is about to be saved / has already been saved.

  Attributes:
    dataset_name: The dataest name.
    creation_time: The dataset's original creation time.
    dataset_creator_version: The original version of the dataset. A value of
      None is only used in datasets that preceded the addition of this field,
      before they are saved to the table.
    get_examples_generator: A callable that supplies the examples' generator.
    key: The (dataset_name, creation_time) key that is used to reference the
      dataset in the DB.
    num_examples: The number of examples contained in the examples' generator.
      Please note that this attribute is available ONLY FOR SAVED DATASETS.
  """
  dataset_name: str
  creation_time: datetime.datetime
  dataset_creator_version: Optional[str]
  get_examples_generator: helpers.Supplier[Iterable[Example]]

  @property
  def key(self) -> tuple[str, datetime.datetime]:
    return self.dataset_name, self.creation_time

  def save(self):
    """Saves this GeneratedDataset to the DB.

    The saved Examples are a base64 encoding (spanner's restrictrion) of the
    Example serializations.
    """
    self._save_dataset_parameters()
    self._save_examples()

  def _save_dataset_parameters(self):
    """Saves dataset's parameters to the GeneratedDatasets table in the db."""
    db = helpers.get_db()
    table_name = 'GeneratedDatasets'
    columns = ('DatasetName', 'CreationTime', 'DatasetCreatorVersion')
    values = [
        (self.dataset_name, self.creation_time, self.dataset_creator_version)
    ]
    with db.batch() as batch:
      batch.insert_or_update(
        table=table_name,
        columns=columns,
        values=values,
      )

  # b/254207953 - Add batching of the insertion mutation.
  # Each serialized example is expected to be far from 10KB, and maximum
  # transaction size is 400MB. Crunching these numbers yields the default value.
  def _save_examples(self):
    """Saves the metadata of the examples to the Examples table in the db."""

    max_mutations = 40000
    columns = ('DatasetName', 'CreationTime', 'ExampleIndex', 'EncodedExample')
    max_examples_in_batch = int(max_mutations / len(columns))
    creation_time = self.creation_time
    dataset_name = self.dataset_name

    examples = self.get_examples_generator()
    index = 0
    db = helpers.get_db()
    for examples_chunk in tqdm.tqdm(
        more_itertools.chunked(examples, max_examples_in_batch),
        unit=' examples',
        unit_scale=max_examples_in_batch,
    ):
      with db.batch() as batch:
        values = []
        for i, example in enumerate(examples_chunk):
          values.append(
              (dataset_name, creation_time, index + i, example.to_db_encoded())
          )
        batch.insert_or_update(
            table=EXAMPLES_TABLE, columns=columns, values=values,
        )
        index += max_examples_in_batch

  @classmethod
  def from_key(
      cls,
      dataset_name: str,
      creation_time: datetime.datetime,
  ) -> Union[GeneratedDataset, None]:
    """Reads the GeneratedDataset from the spanner table.

    Args:
      dataset_name: The name as it appears in the GeneratedDatasets table.
      creation_time: The creation time of the dataset.

    Returns:
      A GeneratedDataset object initialized from the values in spanner, or None
      if the requested dataset does not exist in spanner.
    """

    db = helpers.get_db()
    table_name = 'GeneratedDatasets'
    columns = ('DatasetName', 'CreationTime', 'DatasetCreatorVersion')

    key = (dataset_name, creation_time)
    try:
      with db.snapshot() as snapshot:
        values = snapshot.read(
            table=table_name,
            columns=columns,
            keyset=spanner.KeySet(keys=[key])
        ).one()
    except RowNotFoundError:
      return None

    reader = ExampleReader(
        dataset_name=dataset_name,
        creation_time=creation_time,
        from_index=0,
        until_index=get_number_of_examples(*key)
    )

    # Old datasets might have NULL version, so use the first version for those.
    # Mutate the values retrieved and not the dataclass since it is frozen.
    version_index = columns.index('DatasetCreatorVersion')
    values[version_index] = values[version_index] or '0.0.1'
    return cls(
        *values, get_examples_generator=reader.read
    )  # type: ignore[misc]

  @property
  def num_examples(self) -> int:
    """Returns the number of examples saved for dataset_name, creation_time.

    Raises:
      ValueError: In case the current GeneratedDataset is not yet saved, or
        contains no examples.
    """
    return get_number_of_examples(self.dataset_name, self.creation_time)


@dataclasses.dataclass
class ExampleReader:
  """Reads the examples from the spanner table.

  Attributes:
    dataset_name: The name as it appears in the GeneratedDatasets table.
    creation_time: The creation time of the dataset.
    from_index: The lower bound (inclusive) of `ExampleIndex`s to include.
    until_index: The upper bound (exclusive) of `ExampleIndex`s to include.
  """
  dataset_name: str
  creation_time: datetime.datetime
  from_index: int
  until_index: int

  def read(self) -> Iterable[Example]:
    """Reads the examples from the spanner table.

    Yields:
      An unpopulated Example that matched a row in the spanner table.
    """
    db = helpers.get_db()
    columns = ('EncodedExample',)

    keyset = spanner.KeySet(ranges=[spanner.KeyRange(
      start_closed=[self.dataset_name, self.creation_time, self.from_index],
      end_open=[self.dataset_name, self.creation_time, self.until_index])])
    snapshot = db.batch_snapshot()
    partitions = snapshot.generate_read_batches(
      table=EXAMPLES_TABLE,
      columns=columns,
      keyset=keyset,
    )
    for partition in partitions:
      for row in snapshot.process_read_batch(partition):
        yield Example.from_db_encoded(more_itertools.one(row))


def get_number_of_examples(
    dataset_name: str, creation_time: datetime.datetime) -> int:
  """Returns the number of examples saved for dataset_name, creation_time.

  In case the dataset has not been saved yet, returns 0.

  Args:
    dataset_name: The dataest name.
    creation_time: The dataset's original creation time.
  """
  db = helpers.get_db()
  with db.snapshot() as snapshot:
    results = snapshot.execute_sql(
      """SELECT MAX(DatasetUnpopulatedExamples.ExampleIndex)
      FROM DatasetUnpopulatedExamples
      WHERE DatasetUnpopulatedExamples.DatasetName = @datasetName
            AND DatasetUnpopulatedExamples.CreationTime = @creationTime;""",
      params={'datasetName': dataset_name, 'creationTime': creation_time},
      param_types={'datasetName': spanner.param_types.STRING,
                   'creationTime': spanner.param_types.TIMESTAMP}
    )
    max_index = more_itertools.only(more_itertools.one(results))
  if max_index is None:
    return 0
  return max_index + 1


def get_registered_datasets(
    dataset_name: str,
) -> list[tuple[str, datetime.datetime]]:
  """Returns (dataset_name, creation_time, path_to_sstable) of saved datasets.

  Args:
    dataset_name: The name as it appears in the GeneratedDatasets table.
  """
  db = helpers.get_db()
  table_name = 'GeneratedDatasets'
  cols = ('DatasetName', 'CreationTime')

  keyset = spanner.KeySet(
      ranges=[
          spanner.KeyRange(
              start_closed=[dataset_name], end_closed=[dataset_name]
          )
      ]
  )
  with db.snapshot() as snapshot:
    return list(snapshot.read(table=table_name, columns=cols, keyset=keyset))
