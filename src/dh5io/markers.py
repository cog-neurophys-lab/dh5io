from dhspec.markers import (
    MARKERS_GROUP_NAME,
    MARKERS_DATASET_DTYPE,
)
import numpy as np
import numpy.typing as npt
import h5py
from dh5io.errors import DH5Error
import logging

logger = logging.getLogger(__name__)


def add_marker_to_file(
    file: h5py.File, marker_name: str, timestamps: npt.ArrayLike, replace=True
) -> None:
    if MARKERS_GROUP_NAME not in file:
        file.create_group(MARKERS_GROUP_NAME)
        logger.debug(f"Created '{MARKERS_GROUP_NAME}' group in file {file.filename}")
    markers_group = file[MARKERS_GROUP_NAME]
    if marker_name in markers_group:
        if not replace:
            raise DH5Error(
                f"Marker '{marker_name}' already exists in file {file.filename}"
            )
        del markers_group[marker_name]
        logger.debug(
            f"Replacing existing marker '{marker_name}' in file {file.filename}"
        )
    if (
        not isinstance(timestamps, np.ndarray)
        or timestamps.dtype != MARKERS_DATASET_DTYPE
    ):
        raise DH5Error(
            f"Invalid marker dtype: expected np.int64, got {type(timestamps)}"
        )
    markers_group.create_dataset(
        marker_name, data=np.array(timestamps, dtype=MARKERS_DATASET_DTYPE)
    )


def get_all_markers(file: h5py.File) -> dict[str, np.ndarray]:
    if MARKERS_GROUP_NAME not in file:
        logger.warning(
            f"'{MARKERS_GROUP_NAME}' group not found in file {file.filename}"
        )
        return {}
    markers_group = file[MARKERS_GROUP_NAME]
    markers = {}
    for marker_name, dataset in markers_group.items():
        markers[marker_name] = np.array(dataset, dtype=np.int64)
    return markers


def get_marker_from_file(file: h5py.File, marker_name: str) -> np.ndarray | None:
    if MARKERS_GROUP_NAME not in file:
        logger.warning(
            f"'{MARKERS_GROUP_NAME}' group not found in file {file.filename}"
        )
        return None
    markers_group = file[MARKERS_GROUP_NAME]
    if marker_name not in markers_group:
        logger.warning(f"Marker '{marker_name}' not found in file {file.filename}")
        return None
    return np.array(markers_group[marker_name], dtype=np.int64)


def validate_markers(file: h5py.File) -> None:
    if MARKERS_GROUP_NAME not in file:
        logger.warning(
            f"'{MARKERS_GROUP_NAME}' group not found in file {file.filename}"
        )
        return
    markers_group = file[MARKERS_GROUP_NAME]
    for marker_name, dataset in markers_group.items():
        validate_marker_dataset(marker_name, dataset)


def validate_marker_dataset(marker_name: str, dataset: h5py.Dataset) -> None:
    if not isinstance(dataset, h5py.Dataset) or dataset.dtype != np.int64:
        raise DH5Error(
            f"Marker '{marker_name}' is not a one-dimensional array of 64-bit integers: {dataset.dtype}"
        )
