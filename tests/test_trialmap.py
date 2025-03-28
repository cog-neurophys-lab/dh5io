import pytest
import numpy as np
import h5py
from dh5io.errors import DH5Error
from dhspec.trialmap import TRIALMAP_DATASET_DTYPE, TRIALMAP_DATASET_NAME
from dh5io.trialmap import (
    add_trialmap_to_file,
    get_trialmap_from_file,
    validate_trialmap,
    validate_trialmap_dataset,
)


@pytest.fixture
def mock_h5_file(tmp_path):
    file_path = tmp_path / "test_file.h5"
    with h5py.File(file_path, "w") as f:
        yield f


@pytest.fixture
def valid_trialmap():
    return np.rec.array(
        [
            (1, 101, 0, 1000000000, 2000000000),
            (2, 102, 1, 2000000000, 3000000000),
        ],
        dtype=TRIALMAP_DATASET_DTYPE,
    )


def test_add_trialmap_to_file(mock_h5_file, valid_trialmap):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    assert TRIALMAP_DATASET_NAME in mock_h5_file
    dataset = mock_h5_file[TRIALMAP_DATASET_NAME]
    assert dataset.dtype == TRIALMAP_DATASET_DTYPE
    assert len(dataset) == len(valid_trialmap)
    assert np.array_equal(np.array(dataset), valid_trialmap)


def test_add_trialmap_to_file_invalid_dtype(mock_h5_file):
    invalid_trialmap = np.rec.array(
        [(1, 101)],
        dtype=[("InvalidField", "int32"), ("StimNo", "int32")],
    )
    with pytest.raises(DH5Error, match="Invalid trialmap dtype"):
        add_trialmap_to_file(mock_h5_file, invalid_trialmap)


def test_add_trialmap_to_file_replace(mock_h5_file, valid_trialmap):
    first_trialmap = np.rec.array(
        [(1, 101, 0, 1000000000, 2000000000)], dtype=TRIALMAP_DATASET_DTYPE
    )
    add_trialmap_to_file(mock_h5_file, first_trialmap)
    assert np.array_equal(get_trialmap_from_file(mock_h5_file), first_trialmap)
    with pytest.raises(DH5Error):
        add_trialmap_to_file(mock_h5_file, valid_trialmap, replace=False)
    add_trialmap_to_file(mock_h5_file, valid_trialmap, replace=True)

    assert np.array_equal(get_trialmap_from_file(mock_h5_file), valid_trialmap)


def test_add_trialmap_to_file_no_replace(mock_h5_file, valid_trialmap):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    with pytest.raises(DH5Error):
        add_trialmap_to_file(mock_h5_file, valid_trialmap, replace=False)


def test_get_trialmap_from_file(mock_h5_file, valid_trialmap):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    retrieved_trialmap = get_trialmap_from_file(mock_h5_file)
    assert np.array_equal(retrieved_trialmap, valid_trialmap)


def test_get_trialmap_from_file_no_dataset(mock_h5_file):
    trialmap = get_trialmap_from_file(mock_h5_file)
    assert trialmap is None


def test_validate_trialmap(mock_h5_file, valid_trialmap, caplog):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    validate_trialmap(mock_h5_file)
    assert "TRIALMAP dataset not found" not in caplog.text


def test_validate_trialmap_no_dataset(mock_h5_file, caplog):
    validate_trialmap(mock_h5_file)
    assert "TRIALMAP dataset not found" in caplog.text


def test_validate_trialmap_dataset(mock_h5_file, valid_trialmap):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    dataset = mock_h5_file[TRIALMAP_DATASET_NAME]
    validate_trialmap_dataset(dataset)


def test_validate_trialmap_dataset_invalid(mock_h5_file: h5py.File):
    invalid_dtype = [("InvalidField", "int32")]
    invalid_trialmap = mock_h5_file.create_dataset(
        TRIALMAP_DATASET_NAME, (1,), dtype=invalid_dtype
    )
    with pytest.raises(DH5Error):
        validate_trialmap_dataset(invalid_trialmap)
