import pathlib

from neo.io.basefromrawio import BaseFromRaw
import dh5neo.dh5rawio


class DH5IO(dh5neo.dh5rawio.DH5RawIO, BaseFromRaw):
    """
    Class for reading DAQ-HDF5 (*.dh5) files from the Kreiter lab.
    """

    extensions = ["dh5"]
    mode = "file"

    def __init__(self, filename: str | pathlib.Path):
        dh5neo.DH5RawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
