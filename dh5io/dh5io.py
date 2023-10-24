import pathlib

from neo.io.basefromrawio import BaseFromRaw
from dh5io.dh5rawio import DH5RawIO


class DH5IO(DH5RawIO, BaseFromRaw):
    """
    Class for reading DAQ-HDF5 (*.dh5) files from the Kreiter lab.
    """

    extensions = ["dh5"]
    mode = "file"

    def __init__(self, filename: str | pathlib.Path):
        DH5RawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
