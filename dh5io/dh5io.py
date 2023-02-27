import pathlib

import neo

from neo.io.baseio import BaseIO
from neo.core import (
    Block,
    Segment,
    AnalogSignal,
    IrregularlySampledSignal,
    Epoch,
    Event,
    SpikeTrain,
    ImageSequence,
    ChannelView,
    Group,
)
from neo.io.proxyobjects import BaseProxy
from neo import __version__ as neover


class DH5IO(BaseIO):
    """
    Class for reading DAQ-HDF5 (*.dh5) files from the Kreiter lab.
    """

    extensions = ["dh5"]
    supported_objects = [AnalogSignal, Block, Segment, SpikeTrain, Epoch]
    mode = "file"

    def __init__(self, filename: str | pathlib.Path, mode : str="r"):
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_all_blocks(self, lazy: bool = False):
        if lazy:
            raise Exception("Lazy loading is not supported for NixIO")
        ...

    def read_block(self, index: int | None = None, lazy: bool = False):
        ...
