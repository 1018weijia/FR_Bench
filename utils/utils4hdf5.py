"""
Simple helpers for HDF5 structure.
"""

from typing import Any

import h5py


def print_hdf5_structure(file_path: str) -> None:
    """
    Recursively print an HDF5 file's structure (Group / Dataset names and shapes).

    This is a read-only debug helper and does not modify the file.
    """

    def _print_group(name: str, obj: Any) -> None:
        # If this is a Group, print the group name
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
            for key in obj.keys():
                _print_group(f"{name}/{key}", obj[key])
        # If this is a Dataset, print the dataset name and shape
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, dtype: {obj.dtype}")

    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            _print_group(key, f[key])


__all__ = ["print_hdf5_structure"]

