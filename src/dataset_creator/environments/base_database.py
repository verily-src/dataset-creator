"""A module defining the base class for interacting with a DB."""

import abc
from typing import Any, Iterable, Sequence

import apache_beam as beam


class BaseDatabase(abc.ABC):
  """Base class for interacting with a database."""

  @abc.abstractmethod
  def read(
      self,
      table_name: str,
      columns: Sequence[str] = (),
      filter_key: Any = None,
      range_key_start: Any = None,
      range_key_end: Any = None,
  ) -> Iterable[Sequence[Any]]:
    """Reads data from a table in the database.

    Args:
      table_name: The name of the table to read from.
      columns: The columns to read from the table. Default is to read all
        columns.
      filter_key: The key to filter the rows by. Default is to read
        all rows. Mutually exclusive with range_key_start and range_key_end.
      range_key_start: The start of the range to read from (inclusive). Default
        is to read all rows. Mutually exclusive with filter_key.
      range_key_end: The end of the range to read from (exclusive). Default is
        to read all rows. Mutually exclusive with filter_key.

    Returns:
      An iterable of rows, where each row is a sequence of values, ordered
      according to the given columns argument.
    """

  @abc.abstractmethod
  def query(self, query: str) -> Iterable[Sequence[Any]]:
    """Executes a SQL query on the database.

    Args:
      query: The SQL query to execute.

    Returns:
      An iterable of rows, where each row is a sequence of values.
    """

  @abc.abstractmethod
  def insert(
      self,
      table_name: str,
      columns: Sequence[str],
      rows: Iterable[Sequence[Any]],
  ) -> None:
    """Inserts data into a table in the database.

    Args:
      table_name: The name of the table to insert into.
      columns: The columns to insert into the table.
      rows: The values to insert into the table. Each value is a sequence that
        corresponds to the columns in the same order.
    """

  @abc.abstractmethod
  def create_read_rows_ptransform(
      self, range_key_start: tuple[Any], range_key_end: tuple[Any],
  ) -> beam.PTransform:
    """Creates a PTransform that reads rows from the database.

    Args:
      range_key_start: The start of the range to read from (inclusive).
      range_key_end: The end of the range to read from (exclusive).

    Returns:
      A PTransform that reads rows from the database upon pipeline execution.
    """
