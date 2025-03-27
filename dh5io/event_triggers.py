# This file is part of the DAQ-HDF5 Python package, see LICENSE
"""Event triggers.

Event triggers are a low-level piece of information. They represent the stream
of event triggers in DAQ files, along with the trial descriptor records. Based
on the information from event triggers and trial descriptor records only, it is
not possible to reconstruct a trialmap. But combined with additional information
from the user, these two datasets are used at the early stages of data
processing to produce information which is then written into the Trialmap,
Markers and Intervals. Event triggers and trial descriptor records can also be
used as some intermediate information storage facility during conversion from
other file formats into DAQ-HDF.

Event triggers and trial-descriptor records, imported from DAQ or other file
formats, are likely to contain some sequence errors which increase the
complexity of production of Trialmap, Markers and Intervals. A conversion
program should perform thorough checks of event trigger stream before saving the
results. It is supposed that the information stored in Trialmap, Markers and
Intervals is consistent and has been checked, so that other analysis software
would not need to perform it over again.

Event triggers are stored in DAQ-HDF files in the form of a dataset named EV02
in the root group. EV02 is an array of structures:

Offset  Name    Type
------  ------  -----
0       time    int64
8       event   int32

Each event trigger therefore has a timestamp specified in nanoseconds, and an
encoded event type. Encoding may vary across different experimental setups and
depending on other conditions. No assumptions about encoding are made in
general. Processing and conversion software should receive the information about
the event trigger encoding from other sources than the DAQ-HDF file.

"""

import logging
from dh5io.errors import DH5Error
import h5py
import numpy as np
import numpy.typing as npt

EV_DATASET_NAME = "EV02"
EV_DATASET_DTYPE = np.dtype([("time", np.int64), ("event", np.int32)])
logger = logging.getLogger(__name__)


def add_event_triggers_to_file(
    file: h5py.File,
    timestamps_ns: npt.NDArray[np.int64],  # 1d array of int64
    event_codes: npt.NDArray[np.int32],  # 1d array of int32
    event_triggers_dataset_name: str = EV_DATASET_NAME,
) -> None:
    if EV_DATASET_NAME in file:
        raise DH5Error(
            f"Event triggers dataset {EV_DATASET_NAME} already exists in file {file.filename}"
        )

    if timestamps_ns.shape != event_codes.shape:
        raise DH5Error(
            f"Timestamps and event codes must have the same shape, but have shapes {timestamps_ns.shape} and {event_codes.shape}"
        )

    data = np.empty(len(timestamps_ns), dtype=EV_DATASET_DTYPE)
    data["time"] = timestamps_ns
    data["event"] = event_codes

    file.create_dataset(
        EV_DATASET_NAME,
        data=data,
        dtype=EV_DATASET_DTYPE,
    )


def validate_event_triggers_dataset(dataset: h5py.Dataset) -> None:
    # check for EV02 dataset
    if dataset.dtype != EV_DATASET_DTYPE:
        raise DH5Error(
            f"EV02 dataset must have 2 columns (time, event) with int64 and int32, but has dtype {dataset.dtype}"
        )


def validate_event_triggers(file: h5py.File) -> None:
    # check for EV02 group
    if EV_DATASET_NAME not in file:
        logger.info(f"EV02 dataset not found in file {file.filename}")
        return
    validate_event_triggers_dataset(file[EV_DATASET_NAME])
