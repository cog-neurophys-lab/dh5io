import dh5io.event_triggers as ev
from dh5io.errors import DH5Error
import pytest
import numpy as np
import h5py


def test_add_event_triggers_to_file(tmp_path):
    filename = tmp_path / "test.dh5"
    event_codes = np.array([0, 1, 2], dtype=np.int32)
    timestamps_ns = np.array([1, 2, 3], dtype=np.int64)
    with h5py.File(filename, "w") as h5file:
        ev.add_event_triggers_to_file(h5file, timestamps_ns, event_codes)

    with h5py.File(filename, "r") as h5file:
        assert ev.EV_DATASET_NAME in h5file
        ev.validate_event_triggers(h5file)


def test_add_event_triggers_to_file_existing_dataset(tmp_path):
    filename = tmp_path / "test.dh5"
    event_codes = np.array([0, 1, 2], dtype=np.int32)
    timestamps_ns = np.array([1, 2, 3], dtype=np.int64)
    with h5py.File(filename, "w") as h5file:
        ev.add_event_triggers_to_file(h5file, timestamps_ns, event_codes)
        with pytest.raises(
            DH5Error, match="Event triggers dataset EV02 already exists"
        ):
            ev.add_event_triggers_to_file(h5file, timestamps_ns, event_codes)


def test_add_event_triggers_to_file_mismatched_shapes(tmp_path):
    filename = tmp_path / "test.dh5"
    event_codes = np.array([0, 1], dtype=np.int32)
    timestamps_ns = np.array([1, 2, 3], dtype=np.int64)
    with h5py.File(filename, "w") as h5file:
        with pytest.raises(
            DH5Error, match="Timestamps and event codes must have the same shape"
        ):
            ev.add_event_triggers_to_file(h5file, timestamps_ns, event_codes)


def test_get_event_triggers_from_file(tmp_path):
    filename = tmp_path / "test.dh5"
    event_codes = np.array([0, 1, 2], dtype=np.int32)
    timestamps_ns = np.array([1, 2, 3], dtype=np.int64)
    with h5py.File(filename, "w") as h5file:
        ev.add_event_triggers_to_file(h5file, timestamps_ns, event_codes)

    with h5py.File(filename, "r") as h5file:
        event_triggers = ev.get_event_triggers_from_file(h5file)
        assert event_triggers is not None
        assert len(event_triggers) == 3
        assert np.array_equal(event_triggers["time"], timestamps_ns)
        assert np.array_equal(event_triggers["event"], event_codes)


def test_get_event_triggers_from_file_no_dataset(tmp_path):
    filename = tmp_path / "test.dh5"
    with h5py.File(filename, "w") as h5file:
        event_triggers = ev.get_event_triggers_from_file(h5file)
        assert event_triggers is None


def test_validate_event_triggers_invalid_dtype(tmp_path):
    filename = tmp_path / "test.dh5"
    with h5py.File(filename, "w") as h5file:
        invalid_data = np.array(
            [(1, 0.5)], dtype=[("time", "int64"), ("event", "float64")]
        )
        h5file.create_dataset(ev.EV_DATASET_NAME, data=invalid_data)

    with h5py.File(filename, "r") as h5file:
        with pytest.raises(DH5Error, match="EV02 dataset must have 2 columns"):
            ev.validate_event_triggers(h5file)
