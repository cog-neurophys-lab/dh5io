import pathlib
import pytest
import numpy
import h5py
from dh5io import DH5File, DH5Warning, DH5CalibrationMissingWarning, DH5ChannelsMissingWarning, cont_blocks_start_simultaneously, is_continuous
from dh5io import DH5Error
from dh5io.validation import validate_dh5_file
from dh5io.cont import Cont, create_empty_cont_group_in_file
from dh5io.trialmap import Trialmap
from dh5io.create import create_dh_file

filename = pathlib.Path(__file__).parent / "test.dh5"


@pytest.fixture
def test_file() -> DH5File:
    return DH5File(filename)


dh5 = DH5File("tests/test.dh5", mode="r")


class TestDH5File:
    def test_load(self, test_file: DH5File):
        print(test_file)

    def test_get_version(self, test_file):
        assert test_file.version == 2


class TestDH5FileCont:
    def test_get_cont_groups(self, test_file: DH5File):
        contGroups = test_file.get_cont_groups()
        assert len(contGroups) == 7
        assert all([isinstance(cont, Cont) for cont in contGroups])

    def test_get_cont_group_names(self, test_file: DH5File):
        contNames = test_file.get_cont_group_names()
        assert len(contNames) == 7
        assert contNames == [
            "CONT1",
            "CONT60",
            "CONT61",
            "CONT62",
            "CONT63",
            "CONT64",
            "CONT1001",
        ]

    def test_get_cont_group_ids(self, test_file: DH5File):
        contIds = test_file.get_cont_group_ids()
        assert len(contIds) == 7
        assert contIds == [1, 60, 61, 62, 63, 64, 1001]

    def test_get_cont_group_by_id(self, test_file: DH5File):
        contGroup = test_file.get_cont_group_by_id(1)
        assert isinstance(contGroup, Cont)
        assert contGroup.id == 1
        # expect an DH5Error if the group does not exist
        with pytest.raises(DH5Error):
            test_file.get_cont_group_by_id(99999)

    def test_get_cont_data_by_id(self, test_file: DH5File):
        contData = test_file.get_cont_data_by_id(1)
        assert isinstance(contData, numpy.ndarray)

    def test_get_calibrated_cont_data_by_id(self, test_file: DH5File):
        contData = test_file.get_calibrated_cont_data_by_id(1)
        assert contData.dtype == numpy.float64

    def test_validate_existing_dh5_file(self, test_file: DH5File):
        with pytest.warns(DH5Warning):
            validate_dh5_file(filename)

    @pytest.mark.filterwarnings("ignore::dh5io.errors.DH5Warning")
    @pytest.mark.filterwarnings("ignore::dh5io.errors.DH5ChannelsMissingWarning")
    def test_calibration_missing_warning_has_cont_id(self, test_file: DH5File):
        with pytest.warns(DH5CalibrationMissingWarning) as record:
            validate_dh5_file(filename)
        calibration_warnings = [w for w in record if issubclass(w.category, DH5CalibrationMissingWarning)]
        assert len(calibration_warnings) == 5  # CONT60–64 are missing calibration
        assert all(w.message.cont_id is not None for w in calibration_warnings)
        cont_ids = {w.message.cont_id for w in calibration_warnings}
        assert cont_ids == {60, 61, 62, 63, 64}

    @pytest.mark.filterwarnings("ignore::dh5io.errors.DH5Warning")
    @pytest.mark.filterwarnings("ignore::dh5io.errors.DH5CalibrationMissingWarning")
    def test_channels_missing_warning_has_cont_id(self, test_file: DH5File):
        with pytest.warns(DH5ChannelsMissingWarning) as record:
            validate_dh5_file(filename)
        channels_warnings = [w for w in record if issubclass(w.category, DH5ChannelsMissingWarning)]
        assert len(channels_warnings) == 1  # CONT1001 is missing channels
        assert channels_warnings[0].message.cont_id == 1001


class TestDH5FileSpike:
    # spike groups
    def test_get_spike_groups(self, test_file: DH5File):
        spikeGroups = test_file.get_spike_groups()
        assert len(spikeGroups) == 1
        assert all([isinstance(spike, h5py.Group) for spike in spikeGroups])

    def test_get_spike_group_names(self, test_file: DH5File):
        spikeNames = test_file.get_spike_group_names()
        assert len(spikeNames) == 1
        assert spikeNames == ["SPIKE0"]

    def test_get_spike_group_by_id(self, test_file: DH5File):
        spikeGroup = test_file.get_spike_group_by_id(0)
        assert isinstance(spikeGroup, h5py.Group)
        assert spikeGroup.name == "/SPIKE0"
        assert test_file.get_spike_group_by_id(99999) is None


class TestDH5FileEvent:
    def test_get_events(self, test_file: DH5File):
        events = test_file.get_events_dataset()
        assert events is not None
        assert events.shape == (10460,)
        assert isinstance(events, h5py.Dataset)
        assert events.name == "/EV02"
        for event in events:
            assert len(event) == 2


class TestDH5FileTrialmap:
    def test_get_trialmap(self, test_file: DH5File):
        trialmap = test_file.get_trialmap()
        assert isinstance(trialmap, Trialmap)
        assert trialmap is not None
        assert trialmap.recarray.shape == (385,)
        # assert trialmap.name == "/TRIALMAP"
        assert len(trialmap.recarray.dtype) == 5
        assert trialmap.recarray.dtype.names == (
            "TrialNo",
            "StimNo",
            "Outcome",
            "StartTime",
            "EndTime",
        )

        # test properties
        assert len(trialmap) == 385


class TestContBlocksStartSimultaneously:
    def test_real_file_method(self, test_file: DH5File):
        # All CONT blocks in the test file are from the same recording session
        assert test_file.cont_blocks_start_simultaneously() is True

    def test_real_file_free_function(self):
        assert cont_blocks_start_simultaneously(filename) is True

    def test_single_cont_method(self, tmp_path):
        fname = tmp_path / "single.dh5"
        with create_dh_file(fname) as dh5:
            grp = create_empty_cont_group_in_file(dh5._file, 1, nSamples=100, nChannels=1, sample_period_ns=1_000_000)
            grp["INDEX"][0] = (1_000_000_000, 0)
        with DH5File(fname) as dh5:
            assert dh5.cont_blocks_start_simultaneously() is True

    def test_simultaneous_start_method(self, tmp_path):
        fname = tmp_path / "simultaneous.dh5"
        t0 = 1_000_000_000
        with create_dh_file(fname) as dh5:
            for cid in (1, 2):
                grp = create_empty_cont_group_in_file(dh5._file, cid, nSamples=100, nChannels=1, sample_period_ns=1_000_000)
                grp["INDEX"][0] = (t0, 0)
        with DH5File(fname) as dh5:
            assert dh5.cont_blocks_start_simultaneously() is True

    def test_non_simultaneous_start_method(self, tmp_path):
        fname = tmp_path / "non_simultaneous.dh5"
        with create_dh_file(fname) as dh5:
            grp1 = create_empty_cont_group_in_file(dh5._file, 1, nSamples=100, nChannels=1, sample_period_ns=1_000_000)
            grp1["INDEX"][0] = (1_000_000_000, 0)
            grp2 = create_empty_cont_group_in_file(dh5._file, 2, nSamples=100, nChannels=1, sample_period_ns=1_000_000)
            grp2["INDEX"][0] = (2_000_000_000, 0)
        with DH5File(fname) as dh5:
            assert dh5.cont_blocks_start_simultaneously() is False

    def test_non_simultaneous_start_free_function(self, tmp_path):
        fname = tmp_path / "non_simultaneous2.dh5"
        with create_dh_file(fname) as dh5:
            grp1 = create_empty_cont_group_in_file(dh5._file, 1, nSamples=100, nChannels=1, sample_period_ns=1_000_000)
            grp1["INDEX"][0] = (1_000_000_000, 0)
            grp2 = create_empty_cont_group_in_file(dh5._file, 2, nSamples=100, nChannels=1, sample_period_ns=1_000_000)
            grp2["INDEX"][0] = (2_000_000_000, 0)
        assert cont_blocks_start_simultaneously(fname) is False


class TestIsContinuous:
    def test_real_file_method(self, test_file: DH5File):
        # The test file has CONT blocks with multiple regions
        assert test_file.is_continuous() is False

    def test_real_file_free_function(self):
        assert is_continuous(filename) is False

    def test_single_region_method(self, tmp_path):
        fname = tmp_path / "single_region.dh5"
        with create_dh_file(fname) as dh5:
            grp = create_empty_cont_group_in_file(dh5._file, 1, nSamples=100, nChannels=1, sample_period_ns=1_000_000)
            grp["INDEX"][0] = (1_000_000_000, 0)
        with DH5File(fname) as dh5:
            assert dh5.is_continuous() is True

    def test_multi_region_method(self, tmp_path):
        fname = tmp_path / "multi_region.dh5"
        with create_dh_file(fname) as dh5:
            grp = create_empty_cont_group_in_file(dh5._file, 1, nSamples=100, nChannels=1, sample_period_ns=1_000_000, n_index_items=2)
            grp["INDEX"][0] = (1_000_000_000, 0)
            grp["INDEX"][1] = (2_000_000_000, 50)
        with DH5File(fname) as dh5:
            assert dh5.is_continuous() is False

    def test_multi_region_free_function(self, tmp_path):
        fname = tmp_path / "multi_region2.dh5"
        with create_dh_file(fname) as dh5:
            grp = create_empty_cont_group_in_file(dh5._file, 1, nSamples=100, nChannels=1, sample_period_ns=1_000_000, n_index_items=2)
            grp["INDEX"][0] = (1_000_000_000, 0)
            grp["INDEX"][1] = (2_000_000_000, 50)
        assert is_continuous(fname) is False
