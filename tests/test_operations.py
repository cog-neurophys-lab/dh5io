import pytest
import h5py
import numpy as np
import datetime
from dhspec.operations import (
    OPERATIONS_GROUP_NAME,
    OPERATIONS_OPERATOR_NAME_NAME,
    OPERATIONS_ORIGINAL_FILENAME_NAME,
    OPERATIONS_TOOL_NAME,
    datetime_to_date_array,
)
from dh5io.errors import DH5Error, DH5Warning

from dh5io.operations import (
    add_operation_to_file,
    get_operations_group,
    get_last_operation_index,
    operation_index_from_name,
    validate_operations,
)


@pytest.fixture
def temp_h5_file(tmp_path):
    """Fixture to create a temporary HDF5 file for testing."""
    file_path = tmp_path / "test_file.h5"
    with h5py.File(file_path, "w") as f:
        yield f


def test_add_operation_to_file(temp_h5_file):
    """Test adding an operation to the HDF5 file."""
    tool = "TestTool v1.0"
    operator_name = "Test User"
    operation_name = "TestOperation"
    original_filename = "original_file.h5"
    date = datetime.datetime(2023, 1, 1, 12, 0, 0)

    add_operation_to_file(
        temp_h5_file,
        operation_name,
        tool,
        operator_name,
        date=date,
        original_filename=original_filename,
    )

    operations_group = get_operations_group(temp_h5_file)
    assert operations_group is not None
    assert len(operations_group) == 1

    operation_group_name = list(operations_group.keys())[0]
    assert operation_group_name == "001_TestOperation"

    operation_group = operations_group[operation_group_name]
    assert operation_group.attrs[OPERATIONS_TOOL_NAME] == tool
    assert operation_group.attrs[OPERATIONS_OPERATOR_NAME_NAME] == operator_name
    assert operation_group.attrs[OPERATIONS_ORIGINAL_FILENAME_NAME] == original_filename
    assert np.array_equal(operation_group.attrs["Date"], datetime_to_date_array(date))


def test_get_operations_group(temp_h5_file):
    """Test retrieving the operations group."""
    temp_h5_file.create_group(OPERATIONS_GROUP_NAME)
    operations_group = get_operations_group(temp_h5_file)
    assert operations_group is not None
    assert operations_group.name == f"/{OPERATIONS_GROUP_NAME}"


def test_get_last_operation_index(temp_h5_file):
    """Test retrieving the last operation index."""
    operations_group = temp_h5_file.create_group(OPERATIONS_GROUP_NAME)
    operations_group.create_group("000_FirstOperation")
    operations_group.create_group("001_SecondOperation")

    last_index = get_last_operation_index(temp_h5_file)
    assert last_index == 1


def test_operation_index_from_name():
    """Test extracting the operation index from the operation name."""
    assert operation_index_from_name("001_TestOperation") == 1
    assert operation_index_from_name("000_AnotherOperation") == 0

    with pytest.raises(DH5Error):
        assert operation_index_from_name("InvalidOperation") == 0


def test_validate_operations(temp_h5_file):
    """Test validating the operations group."""
    operations_group = temp_h5_file.create_group(OPERATIONS_GROUP_NAME)
    operations_group.create_group("000_FirstOperation")
    operations_group.create_group("001_SecondOperation")

    validate_operations(temp_h5_file)

    # Test invalid operations group
    del temp_h5_file[OPERATIONS_GROUP_NAME]
    with pytest.raises(DH5Error):
        validate_operations(temp_h5_file)

    # Test non-sequential operation indices
    operations_group = temp_h5_file.create_group(OPERATIONS_GROUP_NAME)
    operations_group.create_group("000_FirstOperation")
    operations_group.create_group("002_ThirdOperation")

    with pytest.warns(DH5Warning):
        validate_operations(temp_h5_file)
