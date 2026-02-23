import datetime
import warnings

import h5py
import h5py.h5t as h5t
import numpy as np
import pytest

from dh5io.cont import create_empty_cont_group_in_file
from dh5io.create import create_dh_file
from dh5io.errors import DH5Warning
from dh5io.hdf5_strings import (
    ascii_str,
    ascii_str_array,
    decode_str_attr,
    write_str_array_attr,
    write_str_attr,
)
from dh5io.operations import add_operation_to_file
from dh5io.wavelet import create_empty_wavelet_group_in_file
from dhspec.cont import CONT_DTYPE_NAME
from dhspec.operations import (
    OPERATIONS_GROUP_NAME,
    OPERATIONS_OPERATOR_NAME_NAME,
    OPERATIONS_ORIGINAL_FILENAME_NAME,
)
from dhspec.wavelet import INDEX_DTYPE, WAVELET_DTYPE_NAME


def test_ascii_str():
    result = ascii_str("hello")
    assert isinstance(result, np.bytes_)
    assert result == b"hello"


def test_decode_str_attr():
    assert decode_str_attr(b"hello") == "hello"
    assert decode_str_attr(np.bytes_(b"hello")) == "hello"
    assert decode_str_attr("hello") == "hello"


def test_ascii_str_array():
    result = ascii_str_array(["a", "bb"])
    assert result.dtype == np.dtype("|S2")
    assert list(result) == [b"a", b"bb"]


def test_operations_strings_are_ascii(tmp_path):
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, "w") as f:
        add_operation_to_file(
            f,
            "TestOp",
            tool="MyTool v1.0",
            operator_name="TestUser",
            original_filename="original.dh5",
            date=datetime.datetime(2023, 1, 1),
        )

    with h5py.File(file_path, "r") as f:
        op_group = f[OPERATIONS_GROUP_NAME]["001_TestOp"]
        assert isinstance(op_group.attrs["Tool"], bytes)
        assert isinstance(op_group.attrs[OPERATIONS_OPERATOR_NAME_NAME], bytes)
        assert isinstance(op_group.attrs[OPERATIONS_ORIGINAL_FILENAME_NAME], bytes)
        assert isinstance(op_group.attrs["dh5io version"], bytes)


def test_cont_strings_are_ascii(tmp_path):
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, "w") as f:
        # Create the CONT_INDEX_ITEM dtype required for CONT groups
        import h5py.h5t as h5t

        tid = h5t.py_create(np.dtype([("time", np.int64), ("offset", np.int64)]))
        tid.commit(f.id, b"CONT_INDEX_ITEM")

        create_empty_cont_group_in_file(
            f,
            cont_group_id=0,
            nSamples=10,
            nChannels=1,
            sample_period_ns=np.int32(1000000),
            name="TestCont",
            comment="A test comment",
        )

    with h5py.File(file_path, "r") as f:
        cont_group = f["CONT0"]
        assert isinstance(cont_group.attrs["Name"], bytes)
        assert isinstance(cont_group.attrs["Comment"], bytes)


def test_cont_default_strings_are_ascii(tmp_path):
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, "w") as f:
        import h5py.h5t as h5t

        tid = h5t.py_create(np.dtype([("time", np.int64), ("offset", np.int64)]))
        tid.commit(f.id, b"CONT_INDEX_ITEM")

        create_empty_cont_group_in_file(
            f,
            cont_group_id=0,
            nSamples=10,
            nChannels=1,
            sample_period_ns=np.int32(1000000),
        )

    with h5py.File(file_path, "r") as f:
        cont_group = f["CONT0"]
        assert isinstance(cont_group.attrs["Name"], bytes)
        assert isinstance(cont_group.attrs["Comment"], bytes)


def test_wavelet_strings_are_ascii(tmp_path):
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, "w") as f:
        create_empty_wavelet_group_in_file(
            f,
            wavelet_group_id=0,
            n_channels=1,
            n_samples=10,
            n_frequencies=5,
            sample_period_ns=np.int32(1000000),
            frequency_axis=np.linspace(1.0, 100.0, 5),
            name="TestWavelet",
            comment="A test comment",
        )

    with h5py.File(file_path, "r") as f:
        wavelet_group = f["WAVELET0"]
        assert isinstance(wavelet_group.attrs["Name"], bytes)
        assert isinstance(wavelet_group.attrs["Comment"], bytes)


def test_wavelet_default_strings_are_ascii(tmp_path):
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, "w") as f:
        create_empty_wavelet_group_in_file(
            f,
            wavelet_group_id=0,
            n_channels=1,
            n_samples=10,
            n_frequencies=5,
            sample_period_ns=np.int32(1000000),
            frequency_axis=np.linspace(1.0, 100.0, 5),
        )

    with h5py.File(file_path, "r") as f:
        wavelet_group = f["WAVELET0"]
        assert isinstance(wavelet_group.attrs["Name"], bytes)
        assert isinstance(wavelet_group.attrs["Comment"], bytes)


def test_create_boards_are_ascii(tmp_path):
    filename = tmp_path / "test.dh5"
    boards = ["board1", "board2"]

    dh5file = create_dh_file(filename, boards=boards)

    with h5py.File(filename, "r") as f:
        boards_attr = f.attrs["BOARDS"]
        for b in boards_attr:
            assert isinstance(b, bytes)


def test_write_str_attr_nullterm_padding(tmp_path):
    """Test that write_str_attr creates attributes with NULLTERM padding."""
    file_path = tmp_path / "test.h5"

    with h5py.File(file_path, "w") as f:
        write_str_attr(f, "test_attr", "hello")

    with h5py.File(file_path, "r") as f:
        # Read the attribute
        attr = f.attrs["test_attr"]
        assert isinstance(attr, bytes)
        assert attr == b"hello"

        # Check the padding type using low-level API
        attr_id = h5py.h5a.open(f.id, b"test_attr")
        dtype = attr_id.get_type()
        padding = dtype.get_strpad()
        assert padding == h5t.STR_NULLTERM


def test_write_str_array_attr_nullterm_padding(tmp_path):
    """Test that write_str_array_attr creates array attributes with NULLTERM padding."""
    file_path = tmp_path / "test.h5"

    with h5py.File(file_path, "w") as f:
        write_str_array_attr(f, "test_array", ["hello", "world", "test"])

    with h5py.File(file_path, "r") as f:
        # Read the attribute
        attr = f.attrs["test_array"]
        assert isinstance(attr, np.ndarray)
        assert len(attr) == 3
        assert list(attr) == [b"hello", b"world", b"test"]

        # Check the padding type using low-level API
        attr_id = h5py.h5a.open(f.id, b"test_array")
        dtype = attr_id.get_type()
        padding = dtype.get_strpad()
        assert padding == h5t.STR_NULLTERM


def test_write_str_array_attr_empty(tmp_path):
    """Test that write_str_array_attr handles empty arrays."""
    file_path = tmp_path / "test.h5"

    with h5py.File(file_path, "w") as f:
        write_str_array_attr(f, "empty_array", [])

    with h5py.File(file_path, "r") as f:
        attr = f.attrs["empty_array"]
        assert isinstance(attr, np.ndarray)
        assert len(attr) == 0


def test_write_str_attr_on_dataset(tmp_path):
    """Test that write_str_attr works on datasets as well as groups."""
    file_path = tmp_path / "test.h5"

    with h5py.File(file_path, "w") as f:
        ds = f.create_dataset("test_dataset", data=np.array([1, 2, 3]))
        write_str_attr(ds, "description", "test dataset")

    with h5py.File(file_path, "r") as f:
        ds = f["test_dataset"]
        attr = ds.attrs["description"]
        assert isinstance(attr, bytes)
        assert attr == b"test dataset"

        # Verify NULLTERM padding
        attr_id = h5py.h5a.open(ds.id, b"description")
        dtype = attr_id.get_type()
        padding = dtype.get_strpad()
        assert padding == h5t.STR_NULLTERM


def test_write_str_array_attr_variable_lengths(tmp_path):
    """Test that write_str_array_attr handles strings of varying lengths."""
    file_path = tmp_path / "test.h5"

    strings = ["a", "bb", "ccc", "dddd"]

    with h5py.File(file_path, "w") as f:
        write_str_array_attr(f, "var_len_array", strings)

    with h5py.File(file_path, "r") as f:
        attr = f.attrs["var_len_array"]
        assert isinstance(attr, np.ndarray)
        assert list(attr) == [b"a", b"bb", b"ccc", b"dddd"]

        # Verify NULLTERM padding
        attr_id = h5py.h5a.open(f.id, b"var_len_array")
        dtype = attr_id.get_type()
        padding = dtype.get_strpad()
        assert padding == h5t.STR_NULLTERM
