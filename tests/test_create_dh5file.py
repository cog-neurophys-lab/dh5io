import numpy as np
import pytest
from dh5io.create import create_dh_file
from dh5io.errors import DH5Warning
from dh5io.validation import validate_dh5_file
import warnings


def test_create_dh5_file(tmp_path):
    filename = tmp_path / "test.dh5"

    version = 45
    boards = ["board1", "board2"]

    dh5file = create_dh_file(filename, file_version=version, boards=boards)

    assert dh5file.version == version
    assert np.array_equal(dh5file.boards, np.array(boards))

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=DH5Warning)
        validate_dh5_file(dh5file.file)

    with pytest.raises(FileExistsError):
        create_dh_file(filename)
