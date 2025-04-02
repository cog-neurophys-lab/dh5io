import pytest
import h5py
import dh5io.cont as cont
import numpy as np
from dh5io.create import create_dh_file
import dh5io


def test_create_empty_cont_group(tmp_path):
    filename = tmp_path / "test.dh5"
    cont_group_name = "test"
    cont_group_id = 100
    sample_period_ns = 1000_000
    calibration = np.array([1.0, 1.0, 3.0])
    nChannels = 3
    nSamples = 123457
    n_index_items = 5
    with create_dh_file(filename) as dh5file:
        cont_group = cont.create_empty_cont_group_in_file(
            dh5file.file,
            cont_group_id=cont_group_id,
            sample_period_ns=sample_period_ns,
            calibration=calibration,
            nChannels=nChannels,
            nSamples=nSamples,
            n_index_items=n_index_items,
            name=cont_group_name,
            signal_type=cont.ContSignalType.LFP,
        )

        assert cont_group.attrs["SamplePeriod"] == sample_period_ns
        assert np.array_equal(cont_group.attrs["Calibration"], calibration)

    with dh5io.DH5File(filename, "r") as dh5file:
        cont_group = dh5file.get_cont_group_by_id(cont_group_id)
        assert cont_group.attrs["SamplePeriod"] == sample_period_ns
        assert cont_group["DATA"].shape == (nSamples, nChannels)
        assert cont_group["INDEX"].shape == (n_index_items,)
        assert np.array_equal(cont_group.attrs["Calibration"], calibration)


def test_create_cont_group_with_data(tmp_path):
    filename = tmp_path / "test.dh5"
    cont_group_name = "test"
    cont_group_id = 100
    sample_period_ns = 1000_000
    calibration = np.array([1.0, 1.0, 3.0])
    n_index_items = 5

    data = np.random.random_integers(-1000, 1000, (100, 3))
    index = cont.create_empty_index_array(n_index_items)
    index[0] = (100, 200)
    index[1] = (300, 400)
    index[2] = (500, 600)
    index[3] = (700, 800)
    index[4] = (900, 1000)
    with create_dh_file(filename) as dh5file:
        cont_group = cont.create_cont_group_from_data_in_file(
            dh5file.file,
            cont_group_id,
            sample_period_ns=sample_period_ns,
            calibration=calibration,
            data=data,
            index=index,
            signal_type=cont.ContSignalType.LFP,
            name=cont_group_name,
        )

        assert cont_group.attrs["SamplePeriod"] == sample_period_ns
        assert cont_group["DATA"].shape == data.shape
        assert cont_group["INDEX"].shape == index.shape
        cont.validate_cont_group(cont_group)

    with dh5io.DH5File(filename, "r") as dh5file:
        cont_group = dh5file.get_cont_group_by_id(cont_group_id)
        assert cont_group.attrs["SamplePeriod"] == sample_period_ns
        assert np.array_equal(cont_group.attrs["Calibration"], calibration)
        assert cont_group["DATA"].shape == data.shape
        assert cont_group["INDEX"].shape == index.shape
        dataset = cont_group["DATA"]
        assert np.array_equal(np.array(cont_group["DATA"]), data)
        assert np.array_equal(np.array(cont_group["INDEX"]), index)
        cont.validate_cont_group(cont_group)
