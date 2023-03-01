import pytest
import dh5io
import pathlib

filename = pathlib.Path(__file__).parent / "test.dh5"


@pytest.fixture
def test_file():
    return dh5io.DH5RawIO(filename)


class TestDH5RawIO:
    def test_load(self, test_file):
        test_file.parse_header()


# class TestDH5IO:

# def test_load(self):
# dh5 = dh5io.DH5IO(filename)
