import numpy as np
import pytest

import dh5io
import dh5io.cont as cont
from dh5io.create import create_dh_file
from dh5io.errors import DH5DataTypeConversionWarning, DH5ChannelsMissingWarning


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
            dh5file._file,
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

        # functional API
        assert cont_group._group.attrs["SamplePeriod"] == sample_period_ns
        assert cont_group._group["DATA"].shape == (nSamples, nChannels)
        assert cont_group._group["INDEX"].shape == (n_index_items,)
        assert np.array_equal(cont_group._group.attrs["Calibration"], calibration)
        assert cont_group.data.shape == (nSamples, nChannels)
        assert cont_group.index.shape == (n_index_items,)
        assert np.array_equal(cont_group.calibration, calibration)
        assert cont_group.sample_period == sample_period_ns
        assert cont_group.signal_type == cont.ContSignalType.LFP

        # object api
        assert cont_group.sample_period == sample_period_ns
        assert np.array_equal(cont_group.calibration, calibration)
        assert cont_group.data.shape == (nSamples, nChannels)
        assert cont_group.index.shape == (n_index_items,)
        assert np.array_equal(cont_group.data, cont_group._group["DATA"])
        assert np.array_equal(cont_group.index, cont_group._group["INDEX"])


def test_create_cont_group_with_data(tmp_path):
    filename = tmp_path / "test.dh5"
    cont_group_name = "test"
    cont_group_id = 100
    sample_period_ns = 1000_000
    calibration = np.array([1.0, 1.0, 3.0])
    n_index_items = 5

    data = np.random.randint(-1000, 1001, (100, 3))
    index = cont.create_empty_index_array(n_index_items)
    index[0] = (100, 200)
    index[1] = (300, 400)
    index[2] = (500, 600)
    index[3] = (700, 800)
    index[4] = (900, 1000)
    with create_dh_file(filename) as dh5file:
        with pytest.warns(DH5DataTypeConversionWarning):
            cont_group = cont.create_cont_group_from_data_in_file(
                dh5file._file,
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
        with pytest.warns(DH5ChannelsMissingWarning):
            cont.validate_cont_group(cont_group)

    with dh5io.DH5File(filename, "r") as dh5file:
        cont_group = dh5file.get_cont_group_by_id(cont_group_id)
        assert cont_group._group.attrs["SamplePeriod"] == sample_period_ns
        assert np.array_equal(cont_group._group.attrs["Calibration"], calibration)
        assert cont_group._group["DATA"].shape == data.shape
        assert cont_group._group["INDEX"].shape == index.shape
        dataset = cont_group._group["DATA"]
        assert np.array_equal(np.array(cont_group._group["DATA"]), data)
        assert np.array_equal(np.array(cont_group._group["INDEX"]), index)
        with pytest.warns(DH5ChannelsMissingWarning):
            cont.validate_cont_group(cont_group._group)

        # hdf5 group contents
        assert cont_group.id == cont_group_id
        assert cont_group.name == cont_group_name
        assert cont_group.sample_period == sample_period_ns
        assert cont_group.data.shape == data.shape
        assert cont_group.index.shape == index.shape
        assert np.array_equal(np.array(cont_group.data), data)
        assert np.array_equal(np.array(cont_group.index), index)
        assert cont_group.signal_type == cont.ContSignalType.LFP

        # object properties
        assert cont_group.sample_period == sample_period_ns
        assert np.array_equal(cont_group.calibration, calibration)
        assert cont_group.data.shape == data.shape
        assert np.array_equal(cont_group.data, data)
        assert cont_group.index.shape == index.shape
        assert np.array_equal(cont_group.index, index)
