"""A module implementing a FileBasedSink to writing example bank shards."""

import os
from typing import Iterable

from apache_beam.coders import coders
from apache_beam.io import filebasedsink
from apache_beam.io import filesystem
# riegeli package is not included in pyproject.toml as it is not available in
# pip yet. Rather, it is built and installed as part of the startup script that
# is configured to all VMs in GCP.
import riegeli  # type: ignore[import]

from dataset_creator import example_lib


def get_positions_file_path(shard_path: str) -> str:
  save_dir = os.path.dirname(shard_path)
  save_basename = os.path.basename(shard_path)
  return os.path.join(save_dir, f'positions-{save_basename}')


class ShardFileSink(filebasedsink.FileBasedSink):
  """A FileBasedSink that writes Examples into shards."""

  def __init__(self, file_path_prefix: str):
    super().__init__(
        file_path_prefix,
        file_name_suffix='.riegeli',
        num_shards=0,
        coder=coders.ToBytesCoder(),
        mime_type='application/octet-stream',
        compression_type=filesystem.CompressionTypes.UNCOMPRESSED
    )

  def open(self, shard_path: str) -> riegeli.RecordWriter:
    """See base class."""
    # Use an uncompressed writer so we can read only a single record when
    # performing random access, instead of reading the entire chunk, then
    # decompressing it so we can get to the desired record.
    return riegeli.RecordWriter(open(shard_path, 'wb'), options='uncompressed')

  def close(self, writer: riegeli.RecordWriter) -> None:
    """See base class."""
    writer.flush()
    writer.close()
    writer.dest.close()

    # Save the written positions, so we can later achieve random access on the
    # shard.
    positions = []
    with open(writer.dest.name, 'rb') as f:
      with riegeli.RecordReader(f) as reader:
        for _ in reader.read_records():
          positions.append(reader.last_pos)
    with open(get_positions_file_path(writer.dest.name), 'wb') as f:
      with riegeli.RecordWriter(f) as positions_writer:
        positions_writer.write_records([pos.to_bytes() for pos in positions])
        positions_writer.flush()

  def write_record(
      self, writer: riegeli.RecordWriter, example: example_lib.Example
  ) -> None:
    writer.write_record(example.to_bytes())

  def finalize_write(
      self, init_result, writer_results, unused_pre_finalize_results
  ) -> Iterable[str]:
    writer_results = sorted(writer_results)
    super_finalize_write_return_value = super().finalize_write(
        init_result, writer_results, unused_pre_finalize_results
    )
    num_shards = len(writer_results)
    for shard_num, original_shard_path in enumerate(writer_results):
      # Use the shard_num to also move the positions file from the temporary
      # directory, and not just the shard itself, which is the responsibility of
      # super().finalize_Write.
      dst_shard_path = self._get_final_name(shard_num, num_shards)
      original_positions_path = get_positions_file_path(original_shard_path)
      dst_positions_path = get_positions_file_path(dst_shard_path)
      os.rename(original_positions_path, dst_positions_path)
    return super_finalize_write_return_value
