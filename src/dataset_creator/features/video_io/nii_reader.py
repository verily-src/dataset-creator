"""A module implementing a NifTi reader."""

import functools

import nibabel as nib

from dataset_creator.features.video_io import nibabel_reader


class NiiReader(nibabel_reader.NibabelReader):
  """A reader for .nii and .nii.gz NifTi files.

  Attributes:
    video_path: The path of the file this reader reads from.

  Raises:
    FileNotFoundError: If check_path and the file does not exist.
    ImageFileError: If check_path and the file is not a valid NifTi file.
    IndexError: If the z_slice is invalid.
  """

  @functools.cached_property
  def fps(self) -> float:
    """Returns the fps of the NifTi file."""
    if not isinstance(self._nib_img, nib.Nifti1Image):
      raise RuntimeError('This operation requires a Nifti image!')
    t_unit = self._nib_img.header.get_xyzt_units()[-1]
    fps_multiplier = {'sec': 1, 'msec': 1000}
    return float(
        fps_multiplier[t_unit] / self._nib_img.header.get_zooms()[-1]
    )
