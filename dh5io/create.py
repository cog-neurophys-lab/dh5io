import logging
from dh5io.dh5file import DH5File
import pathlib
import os.path
import h5py
import h5py.h5t as h5t
from dh5io.validation import validate_dh5_file
from dh5io.operations import add_operation_to_file
import numpy

logger = logging.getLogger(__name__)


def create_dh_file(filename: str | pathlib.Path) -> DH5File:
    if os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists.")

    h5file = h5py.File(filename, mode="w-")
    h5file.attrs["FILEVERSION"] = 2

    tid = h5t.py_create(numpy.dtype([("time", numpy.int64), ("offset", numpy.int64)]))
    tid.commit(h5file.id, b"CONT_INDEX_ITEM")

    add_operation_to_file(h5file, "create_file", tool="dh5io", id=0)

    validate_dh5_file(h5file)

    return h5file
