import pytest
import dh5io

filename = "test.dh5"

class TestDH5RawIO:
    dh5 = dh5io.DH5RawIO(filename)

class TestDH5IO:
    dh5 = dh5io.DH5IO(filename)




