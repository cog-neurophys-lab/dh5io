import pathlib
import pytest
from dh5io import DH5File

filename = pathlib.Path(__file__).parent / "test.dh5"

@pytest.fixture
def test_file() -> DH5File:
    return DH5File(filename)

dh5 = DH5File("tests/test.dh5")

class TestDH5File:
    def test_load(self, test_file: DH5File):
        print(test_file)
