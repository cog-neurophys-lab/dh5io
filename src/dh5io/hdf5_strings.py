import h5py
import h5py.h5t as h5t
import numpy as np


def ascii_str(value: str) -> np.bytes_:
    """Convert a string to fixed-length ASCII for HDF5 attributes.

    Note: Returns numpy bytes which will be written with NULLPAD by default.
    For NULLTERM padding, use write_str_attr() instead.
    """
    return np.bytes_(value)


def ascii_str_array(values: list[str]) -> np.ndarray:
    """Convert a list of strings to a fixed-length ASCII array for HDF5 attributes.

    Note: Returns numpy bytes array which will be written with NULLPAD by default.
    For NULLTERM padding, use write_str_array_attr() instead.
    """
    return np.array(values, dtype="S")


def decode_str_attr(value: str | bytes | np.bytes_) -> str:
    """Decode an HDF5 string attribute value to str.

    Fixed-length ASCII attributes are returned by h5py as bytes; this
    normalises them to str so callers always receive a consistent type.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("ascii")
    return value


def _create_nullterm_str_type(length: int) -> h5t.TypeStringID:
    """Create a fixed-length HDF5 string type with NULLTERM padding.

    Args:
        length: The fixed length in bytes for the string type.

    Returns:
        An HDF5 type ID for a fixed-length string with NULLTERM padding.
    """
    str_type = h5t.C_S1.copy()
    str_type.set_size(length)
    str_type.set_strpad(h5t.STR_NULLTERM)
    str_type.set_cset(h5t.CSET_ASCII)
    return str_type


def write_str_attr(h5obj: h5py.Group | h5py.Dataset, name: str, value: str) -> None:
    """Write a string attribute with NULLTERM padding.

    This function writes a fixed-length ASCII string attribute to an HDF5
    group or dataset using H5T_STR_NULLTERM padding instead of the default
    H5T_STR_NULLPAD.

    Args:
        h5obj: The HDF5 group or dataset to write the attribute to.
        name: The attribute name.
        value: The string value to write.
    """
    # Convert to bytes and get length
    value_bytes = value.encode("ascii")
    length = len(value_bytes) + 1  # +1 for null terminator

    # Create the HDF5 type with NULLTERM padding
    str_type = _create_nullterm_str_type(length)

    # Create space for scalar
    space = h5py.h5s.create(h5py.h5s.SCALAR)

    # Create and write the attribute
    attr_id = h5py.h5a.create(h5obj.id, name.encode("ascii"), str_type, space)
    # Write the null-terminated string data
    data = np.frombuffer(value_bytes + b"\x00", dtype=np.uint8)
    attr_id.write(data, mtype=str_type)
    attr_id.close()


def write_str_array_attr(
    h5obj: h5py.Group | h5py.Dataset, name: str, values: list[str]
) -> None:
    """Write a string array attribute with NULLTERM padding.

    This function writes a fixed-length ASCII string array attribute to an HDF5
    group or dataset using H5T_STR_NULLTERM padding instead of the default
    H5T_STR_NULLPAD.

    Args:
        h5obj: The HDF5 group or dataset to write the attribute to.
        name: The attribute name.
        values: The list of string values to write.
    """
    if not values:
        # Handle empty array case
        h5obj.attrs[name] = np.array([], dtype="S1")
        return

    # Convert all strings to bytes
    values_bytes = [v.encode("ascii") for v in values]

    # Find the maximum length (including null terminator)
    max_len = max(len(b) for b in values_bytes) + 1

    # Create the HDF5 type with NULLTERM padding
    str_type = _create_nullterm_str_type(max_len)

    # Create the data array, each element padded to max_len and null-terminated
    n_elements = len(values)
    data = np.zeros((n_elements, max_len), dtype=np.uint8)
    for i, b in enumerate(values_bytes):
        data[i, : len(b)] = list(b)
        data[i, len(b)] = 0  # null terminator

    # Create space for the array
    space = h5py.h5s.create_simple((n_elements,))

    # Create and write the attribute
    attr_id = h5py.h5a.create(h5obj.id, name.encode("ascii"), str_type, space)
    attr_id.write(data, mtype=str_type)
    attr_id.close()
