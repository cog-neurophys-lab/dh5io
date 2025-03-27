import pytest
import h5py
import dh5io.cont as cont
import numpy as np
from dh5io.create import create_dh5_file
import dh5io


def test_create_empty_cont_group(tmp_path):
    filename = tmp_path / "test.dh5"
    cont_group_name = "test"
    cont_group_id = 100
    sample_period_ns = 1000_000
    calibration = 0.0001
    nChannels = 3
    nSamples = 123457
    n_index_items = 5
    with create_dh5_file(filename) as dh5file:
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
        assert cont_group.attrs["Calibration"] == np.float64(calibration)

    with dh5io.DH5File(filename, "r") as dh5file:
        cont_group = dh5file.get_cont_group_by_id(cont_group_id)
        assert cont_group.attrs["SamplePeriod"] == sample_period_ns
        assert cont_group["DATA"].shape == (nSamples, nChannels)
        assert cont_group["INDEX"].shape == (n_index_items,)


@pytest.mark.skip(reason="Not implemented yet")
def test_add_cont_group_to_file(tmp_path):
    filename = tmp_path / "test.dh5"

    dh5file = create_dh5_file(filename)

    data = [1, 2, 3]
    index = [0, 1, 2]
    sample_period_ns = 1
    cont_group_name = "test"
    cont_group_id = 0
    calibration = "1 V"
    channels = [0, 1, 2]

    cont_group = cont.create_cont_group_from_data_in_file(
        dh5file.file,
        cont_group_id=cont_group_id,
        cont_group_name=cont_group_name,
        data=data,
        index=index,
        sample_period_ns=sample_period_ns,
        calibration=calibration,
        channels=channels,
    )

    assert cont_group.attrs["SamplePeriod"] == sample_period_ns
    assert cont_group.attrs["Calibration"] == calibration
    assert (cont_group.attrs["Channels"] == channels).all()
    assert (cont_group["DATA"][:] == data).all()
    assert (cont_group["INDEX"][:] == index).all()

    with pytest.raises(FileExistsError):
        dh5file.add_cont_group(
            cont_group_id=cont_group_id,
            cont_group_name=cont_group_name,
            data=data,
            index=index,
            sample_period_ns=sample_period_ns,
            calibration=calibration,
            channels=channels,
        )
