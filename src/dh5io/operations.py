"""Processing history in Operations group of DAQ-HDF5 files.

During the analysis of data, contents of a DAQ-HDF file may change. For
example, user can apply a filter to the signal data. For each such
operation, a history record is added to a DAQ-HDF file, in order to let
the user to trace these operations back in time and check a file's
condition when necessary.

For each operation, typically, the program name and version number is
written, as well as the operator's name, and other parameters relevant
to a specific operation.

Processing history is stored in DAQ-HDF files under the group named
`/Operations`. Usually processing history contains at least one entry
which describes how a file was created. Software tools which change
DAQ-HDF files should add history entries.

Each history entry is stored in a subgroup in the `/Operations` group.
The name of this subgroup should be given as follows:

```
nnn_OperationName
```

Where `nnn` is a 3-digit number with leading zeros if this number is less
than 100. Numeration starts from 0. When a new history entry is added,
it gets a number which equals the number of the last existing history
entry plus one.

Numeration is necessary to preserve the order of history entries,
because group and dataset names in HDF5-files are automatically sorted.

Any information about a particular operation performed on the file is
stored as attributes of this operation's subgroup. At the moment,
datasets are not allowed here.

There are no strict definitions about what information should be written
about each operation. However, typically the following attributes are
written at all cases:

- **Tool** (`string` scalar) – Title and version of the software tool used to
perform named operation;

- **Operator name** (`string` scalar) – Name of the person who initiated and
controlled the named operation; preferably full name;

- **Date** (`struct` scalar) – Date and time when this operation was
performed:

    | Offset | Name | Type |
    |-----|--------|-------|
    | 0   | Year   | `int16` |
    | 2   | Month  | `int8`  |
    | 3   | Day    | `int8`  |
    | 4   | Hour   | `int8`  |
    | 5   | Minute | `int8`  |
    | 6   | Second | `int8`  |

- **Original file name** (`string` scalar) – If the processing involved
creation of a new DAQ-HDF file and copying some of the initial file's
contents, instead of just modifying the old file, this attribute is used
to specify the name of the initial file. For example, when a DAQ-file is
converted into a DAQ-HDF file, original DAQ-filename is written here.
Preferably, original file name should be specified exactly as provided
to the processing tool, which means no truncation of the file path.

It is recommended that data processing tools always write all important
parameters provided to these tools from their operators, unless storing
such information would dramatically increase the size of a DAQ-HDF file.

If a reversible modification was performed on a DAQ-HDF file and then it
was undone, history entries should be neither deleted nor modified. An
undo entry must be added to the processing history instead.

"""

import datetime
import logging
import pathlib
import h5py
import h5py.h5t
from dh5io.ensure_h5py_file import ensure_h5py_file
from dh5io.errors import DH5Error, DH5Warning
from dhspec.operations import (
    OPERATIONS_DATE_NAME,
    OPERATIONS_GROUP_NAME,
    OPERATIONS_OPERATOR_NAME_NAME,
    OPERATIONS_ORIGINAL_FILENAME_NAME,
    datetime_to_date_array,
)
import warnings
import getpass
import numpy as np
from dh5io.version import get_version

logger = logging.getLogger(__name__)


@ensure_h5py_file
def add_operation_to_file(
    file: h5py.File,
    new_operation_group_name: str,
    tool: str,
    operator_name: str | None = None,
    id: int | None = None,
    date: datetime.datetime = datetime.datetime.now(),
    original_filename: str | pathlib.Path | None = None,
):
    if id is None:
        last_index = get_last_operation_index(file)
        if last_index is None:
            last_index = 0
        id = last_index + 1

    new_operation_group_name = f"{id:03}_{new_operation_group_name}"
    operations_group = get_operations_group(file)
    if operations_group is None:
        operations_group = file.create_group(OPERATIONS_GROUP_NAME)

    new_operation_group = operations_group.create_group(new_operation_group_name)

    new_operation_group.attrs["Tool"] = tool

    if operator_name is None:
        operator_name = getpass.getuser()

    # write attrs to file
    new_operation_group.attrs[OPERATIONS_OPERATOR_NAME_NAME] = operator_name

    if original_filename is not None:
        new_operation_group.attrs[OPERATIONS_ORIGINAL_FILENAME_NAME] = str(
            original_filename
        )

    new_operation_group.attrs["dh5io version"] = get_version()

    new_operation_group.attrs[OPERATIONS_DATE_NAME] = datetime_to_date_array(date)

    logger.info(f"Added operation {new_operation_group_name} to file {file.filename}")


def get_operations_group(file: h5py.File) -> h5py.Group | None:
    return file.get(OPERATIONS_GROUP_NAME, default=None)


@ensure_h5py_file
def get_last_operation_index(file: h5py.File) -> int | None:
    operations = get_operations_group(file)
    if operations is None:
        return None
    return operation_index_from_name(list(operations.keys())[-1])


def operation_index_from_name(operation_name: str) -> int:
    strId = operation_name.split("_")[0]
    if len(strId) != 3:
        warnings.warn(
            message=f"Operation index {strId} of operation {operation_name} is not a three digit number",
            category=DH5Warning,
        )
    try:
        id = int(strId)
    except ValueError:
        raise DH5Error(
            f"Operation index {strId} of operation {operation_name} is not a valid integer",
        )
    return id


@ensure_h5py_file
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
