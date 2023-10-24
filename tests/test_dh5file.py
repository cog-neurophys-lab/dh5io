import pathlib
import pytest
import numpy
import h5py
from dh5io import DH5File
from dh5io.dh5file import DH5Error

filename = pathlib.Path(__file__).parent / "test.dh5"

@pytest.fixture
def test_file() -> DH5File:
    return DH5File(filename)

dh5 = DH5File("tests/test.dh5")

class TestDH5File:
    def test_load(self, test_file: DH5File):
        print(test_file)

    def test_get_version(self, test_file):
        assert test_file.get_version() == 2
    
    def test_get_cont_groups(self, test_file):
        contGroups = test_file.get_cont_groups()
        assert len(contGroups) == 7
        assert all([isinstance(cont, h5py.Group) for cont in contGroups])

    def test_get_cont_group_names(self, test_file):
        contNames = test_file.get_cont_group_names()
        assert len(contNames) == 7
        assert contNames == ['CONT1', 'CONT1001', 'CONT60', 'CONT61', 'CONT62', 'CONT63', 'CONT64']
    
    def test_get_cont_group_ids(self, test_file):
        contIds = test_file.get_cont_group_ids()
        assert len(contIds) == 7
        assert contIds == [1, 1001, 60, 61, 62, 63, 64]

    def test_get_cont_group_by_id(self, test_file):
        contGroup = test_file.get_cont_group_by_id(1)
        assert isinstance(contGroup, h5py.Group)
        assert contGroup.name == "/CONT1"
        # expect an DH5Error if the group does not exist
        with pytest.raises(DH5Error):
            test_file.get_cont_group_by_id(99999)

    def test_get_cont_data_by_id(self, test_file):
        contData = test_file.get_cont_data_by_id(1)
        assert isinstance(contData, numpy.ndarray)

    def test_get_calibrated_cont_data_by_id(self, test_file):
        contData = test_file.get_calibrated_cont_data_by_id(1)
        assert contData.dtype == numpy.float64
