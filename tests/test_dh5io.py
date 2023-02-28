import pytest
import dh5io
import pathlib

filename = pathlib.Path(__file__).parent / "test.dh5"


class TestDH5RawIO:


    def test_load(self):
        dh5 = dh5io.DH5RawIO(filename)

class TestDH5IO:

    def test_load(self):
        dh5 = dh5io.DH5IO(filename)




