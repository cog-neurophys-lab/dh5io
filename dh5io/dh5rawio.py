import neo
from neo.rawio.baserawio import BaseRawIO

class DH5RawIO(BaseRawIO):
    """
    Class for reading DAQ-HDF5 (*.dh5) files from the Kreiter lab.
    """