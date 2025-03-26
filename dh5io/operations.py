import datetime
import pathlib
import h5py
import h5py.h5t
from dh5io.errors import DH5Error, DH5Warning
import warnings
import getpass
import numpy as np
from dh5io.version import get_version


OPERATIONS_GROUP_NAME = "Operations"


def add_operation_to_file(
    file: str | pathlib.Path | h5py.File,
    new_operation_group_name: str,
    tool: str,
    operator_name: str | None = None,
    id: int | None = None,
    date: datetime.datetime = datetime.datetime.now(),
    original_filename: str | pathlib.Path | None = None,
):
    if isinstance(file, str) | isinstance(file, pathlib.Path):
        file = h5py.File(file, "w")

    if not isinstance(file, h5py.File):
        raise TypeError("file argument must be string or h5py.File")

    if id is None:
        last_index = get_last_operation_index(file)
        if last_index is None:
            last_index = 0
        id = last_index + 1

    new_operation_group_name = f"{id:3}_{new_operation_group_name}"
    operations_group = get_operations_group(file)
    if operations_group is None:
        operations_group = file.create_group(OPERATIONS_GROUP_NAME)

    new_operation_group = operations_group.create_group(new_operation_group_name)

    new_operation_group.attrs["Tool"] = tool

    if operator_name is None:
        operator_name = getpass.getuser()

    # write attrs to file
    new_operation_group.attrs["Operator name"] = operator_name

    if original_filename is not None:
        new_operation_group.attrs["original_filename"] = str(original_filename)

    new_operation_group.attrs["dh5io version"] = get_version()

    date_dtype = np.dtype(
        [
            ("Year", np.int64),
            ("Month", np.int8),
            ("Day", np.int8),
            ("Hour", np.int8),
            ("Minute", np.int8),
            ("Second", np.int8),
        ]
    )
    current_datetime = np.array(
        (date.year, date.month, date.day, date.hour, date.minute, date.second),
        dtype=date_dtype,
    )
    new_operation_group.attrs["Date"] = current_datetime


def get_operations_group(file: h5py.File) -> h5py.Group | None:
    return file.get(OPERATIONS_GROUP_NAME, default=None)


def get_last_operation_index(file: h5py.File) -> int | None:
    operations = get_operations_group(file)
    if operations is None:
        return None
    return operation_index_from_name(operations.keys()[-1])


def operation_index_from_name(operation_name: str) -> int:
    strId = operation_name.split("_")[0]
    if len(strId) != 3:
        warnings.warn(
            message=f"Operation index {strId} of operation {operation_name} is not a three digit number",
            category=DH5Warning,
        )
    return int(strId)


def validate_operations(file: h5py.File):
    if OPERATIONS_GROUP_NAME not in file:
        raise DH5Error(f"No operations defined in {file.filename}")

    operations: h5py.Group = file[OPERATIONS_GROUP_NAME]
    if not isinstance(operations, h5py.Group):
        raise DH5Error(f"Operations in {file.filename} are not a valid HDF5 group")

    for id, op in enumerate(operations):
        if not isinstance(operations[op], h5py.Group):
            raise DH5Error(
                f"Operation {op.name} in {file.filename} is not a valid HDF5 group"
            )

        if id != operation_index_from_name(op):
            warnings.warn(DH5Warning("Operation indices are not numbered sequentially"))
