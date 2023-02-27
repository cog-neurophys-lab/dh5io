import pytest
import dh5io

def test_dh5io():
    dh5 = dh5io.DH5IO("")
    rawdh5 = dh5io.DH5RawIO("")
