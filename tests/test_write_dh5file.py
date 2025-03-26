import pytest
from dh5io.create import create_dh5_file
from dh5io.validation import validate_dh5_file


def test_create_dh5_file(tmp_path):
    filename = tmp_path / "test.dh5"

    dh5file = create_dh5_file(filename)

    validate_dh5_file(dh5file)

    with pytest.raises(FileExistsError):
        create_dh5_file(filename)
