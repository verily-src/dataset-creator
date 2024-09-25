"""A GCP implementation for a database using Cloud Spanner."""

import functools
from typing import Any, Iterable, Sequence

import apache_beam as beam
from apache_beam.io.gcp.experimental import spannerio
from google.cloud import spanner  # type: ignore[attr-defined]
from google.cloud.spanner_v1 import database  # type: ignore[import]
import more_itertools
import tqdm

from dataset_creator.environments import base_database
from dataset_creator.environments.gcp import gcp_utils

_SPANNER_INSTANCE_NAME = 'datasetcreator'
_SPANNER_DB_NAME = 'dataset_creator_db'


class SpannerDatabase(base_database.BaseDatabase):
  """An implementation of a BaseDatabase based on Cloud spanner."""

  @functools.cached_property
  def _db(self) -> database.Database:
    spanner_client = spanner.Client(project=gcp_utils.get_project_id())
    instance = spanner_client.instance(_SPANNER_INSTANCE_NAME)
    return instance.database(_SPANNER_DB_NAME)

  def read(
      self,
      table_name: str,
      columns: Sequence[str] = (),
      filter_key: Any = None,
      range_key_start: Any = None,
      range_key_end: Any = None,
  ) -> Iterable[Sequence[Any]]:
    """See base class."""
    if range_key_start or range_key_end:
      keyset = spanner.KeySet(
          ranges=[
              spanner.KeyRange(
                  start_closed=range_key_start, end_open=range_key_end
              )
          ]
      )
    else:
      keyset = spanner.KeySet(keys=[filter_key])
    snapshot = self._db.batch_snapshot()
    partitions = snapshot.generate_read_batches(
        table=table_name,
        columns=columns,
        keyset=keyset,
    )
    for partition in partitions:
      yield from snapshot.process_read_batch(partition)

  def query(self, query: str) -> Iterable[Sequence[Any]]:
    """See base class."""
    with self._db.snapshot() as snapshot:
      return snapshot.execute_sql(query)

  # Each row is expected to be far from 10KB, and maximum transaction size is
  # 400MB. Crunching these numbers yields the default batching value.
  def insert(
      self,
      table_name: str,
      columns: Sequence[str],
      rows: Iterable[Sequence[Any]],
  ) -> None:
    """See base class."""
    max_mutations = 40000
    max_rows_in_batch = int(max_mutations / len(columns))
    for rows_chunk in tqdm.tqdm(
        more_itertools.chunked(rows, max_rows_in_batch),
        unit=' rows',
        unit_scale=max_rows_in_batch,
    ):
      with self._db.batch() as batch:
        batch.insert_or_update(
            table=table_name, columns=columns, values=rows_chunk,
        )

  def create_read_rows_ptransform(
      self, range_key_start: tuple[Any], range_key_end: tuple[Any],
  ) -> beam.PTransform:
    """See base class."""
    db = self._db

    class _ReadFromSpanner(beam.PTransform):
      def expand(self, input_or_inputs: beam.PCollection) -> beam.PCollection:
        # pylint: disable-next=import-outside-toplevel
        from dataset_creator import generated_dataset

        *_, instance_id, _, database_id = db.name.split('/')
        key_range = spanner.KeyRange(
            start_closed=range_key_start, end_open=range_key_end,
        )
        return (
            input_or_inputs
            | 'ReadFromSpanner' >> spannerio.ReadFromSpanner(
                project_id=gcp_utils.get_project_id(),
                instance_id=instance_id,
                database_id=database_id,
                table=generated_dataset.EXAMPLES_TABLE,
                columns=['EncodedExample'],
                keyset=spannerio.KeySet(ranges=[key_range]),
            )
        )

    return _ReadFromSpanner()
