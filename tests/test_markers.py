import pytest
import numpy as np
import h5py
from dh5io.errors import DH5Error
from dhspec.markers import MARKERS_GROUP_NAME, MARKERS_DATASET_DTYPE

from dh5io.markers import (
    add_marker_to_file,
    get_all_markers,
    get_marker_from_file,
    validate_markers,
    validate_marker_dataset,
)


@pytest.fixture
def mock_h5_file(tmp_path):
    file_path = tmp_path / "test_file.h5"
    with h5py.File(file_path, "w") as f:
        yield f


@pytest.fixture
def valid_markers():
    return {
        "marker1": np.array([1000000000, 2000000000], dtype=np.int64),
        "marker2": np.array([3000000000], dtype=np.int64),
    }


def test_add_markers_to_file(mock_h5_file, valid_markers):
    name = "marker1"
    times = valid_markers["marker1"]
    add_marker_to_file(mock_h5_file, marker_name="marker1", timestamps=times)
    assert MARKERS_GROUP_NAME in mock_h5_file
    group = mock_h5_file[MARKERS_GROUP_NAME]
    assert name in group
    dataset = group[name]
    assert dataset.dtype == MARKERS_DATASET_DTYPE
    assert np.array_equal(dataset, times)


def test_add_markers_to_file_invalid_dtype(mock_h5_file):
    invalid_markers = {"marker1": np.array([1.5, 2.5], dtype=np.float64)}
    with pytest.raises(DH5Error, match="Invalid marker dtype"):
        add_marker_to_file(
            mock_h5_file, marker_name="marker1", timestamps=invalid_markers["marker1"]
        )


# def test_add_markers_to_file_replace(mock_h5_file, valid_markers):
#     first_markers = {"marker1": np.array([1000000000], dtype=np.int64)}
#     add_marker_to_file(mock_h5_file, first_markers)
#     assert np.array_equal(
#         get_marker_from_file(mock_h5_file)["marker1"], first_markers["marker1"]
#     )
#     with pytest.raises(DH5Error):
#         add_marker_to_file(mock_h5_file, valid_markers, replace=False)
#     add_marker_to_file(mock_h5_file, valid_markers, replace=True)

#     retrieved_markers = get_marker_from_file(mock_h5_file)
#     for name, times in valid_markers.items():
#         assert np.array_equal(retrieved_markers[name], times)


# def test_add_markers_to_file_no_replace(mock_h5_file, valid_markers):
#     add_marker_to_file(mock_h5_file, valid_markers)
#     with pytest.raises(DH5Error):
#         add_marker_to_file(mock_h5_file, valid_markers, replace=False)

#         def test_get_all_markers(mock_h5_file, valid_markers):
#             add_marker_to_file(mock_h5_file, valid_markers)
#             retrieved_markers = get_all_markers(mock_h5_file)
#             assert isinstance(retrieved_markers, dict)
#             assert len(retrieved_markers) == len(valid_markers)
#             for name, times in valid_markers.items():
#                 assert name in retrieved_markers
#                 assert np.array_equal(retrieved_markers[name], times)

#         def test_get_all_markers_no_group(mock_h5_file):
#             markers = get_all_markers(mock_h5_file)
#             assert markers == {}


# def test_get_markers_from_file(mock_h5_file, valid_markers):
#     add_marker_to_file(mock_h5_file, valid_markers)
#     retrieved_markers = get_marker_from_file(mock_h5_file)
#     for name, times in valid_markers.items():
#         assert np.array_equal(retrieved_markers[name], times)


# def test_get_markers_from_file_no_group(mock_h5_file):
#     markers = get_marker_from_file(mock_h5_file)
#     assert markers is None


# def test_validate_markers(mock_h5_file, valid_markers, caplog):
#     add_marker_to_file(mock_h5_file, valid_markers)
#     validate_markers(mock_h5_file)
#     assert "Markers group not found" not in caplog.text


# def test_validate_markers_no_group(mock_h5_file, caplog):
#     validate_markers(mock_h5_file)
#     assert "Markers group not found" in caplog.text


# def test_validate_marker_dataset(mock_h5_file, valid_markers):
#     add_marker_to_file(mock_h5_file, valid_markers)
#     group = mock_h5_file[MARKERS_GROUP_NAME]
#     for dataset in group.values():
#         validate_marker_dataset(dataset)


# def test_validate_marker_dataset_invalid(mock_h5_file):
#     invalid_dtype = np.array([1.5, 2.5], dtype=np.float64)
#     group = mock_h5_file.create_group(MARKERS_GROUP_NAME)
#     group.create_dataset("invalid_marker", data=invalid_dtype)
#     with pytest.raises(DH5Error):
#         validate_marker_dataset(group["invalid_marker"])
