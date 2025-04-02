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


def create_dh_file(
    filename: str | pathlib.Path,
    overwrite=False,
    file_version: int = 2,
    boards: list[str] = [],
) -> DH5File:
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists.")

    dh5File = DH5File(filename, mode="w")
    h5file = dh5File.file
    h5file.attrs["FILEVERSION"] = file_version

    h5file.attrs["BOARDS"] = numpy.array(
        boards, dtype=h5py.string_dtype(encoding="utf-8")
    )

    tid = h5t.py_create(numpy.dtype([("time", numpy.int64), ("offset", numpy.int64)]))
    tid.commit(h5file.id, b"CONT_INDEX_ITEM")

    add_operation_to_file(h5file, "create_file", tool="dh5io", id=0)

    validate_dh5_file(h5file)

    return dh5File
