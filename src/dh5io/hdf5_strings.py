import numpy as np


def ascii_str(value: str) -> np.bytes_:
    """Convert a string to fixed-length ASCII for HDF5 attributes."""
    return np.bytes_(value)


def ascii_str_array(values: list[str]) -> np.ndarray:
    """Convert a list of strings to a fixed-length ASCII array for HDF5 attributes."""
    return np.array(values, dtype="S")


def decode_str_attr(value: str | bytes | np.bytes_) -> str:
    """Decode an HDF5 string attribute value to str.

    Fixed-length ASCII attributes are returned by h5py as bytes; this
    normalises them to str so callers always receive a consistent type.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("ascii")
    return value
