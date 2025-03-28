import dh5io.event_triggers as ev
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
