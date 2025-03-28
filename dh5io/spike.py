from dataclasses import dataclass
import h5py
import numpy as np
from dh5io.ensure_h5py_file import ensure_h5py_file
from dhspec.cont import CalibrationType
from dhspec.spike import (
    SPIKE_PREFIX,
    DATA_DATASET_NAME,
    INDEX_DATASET_NAME,
    CLUSTER_INFO_DATASET_NAME,
    SPIKE_PARAMS_DTYPE,
    SPIKE_CHANNELS_DTYPE,
    SpikeParams,
    spike_name_from_id,
    spike_id_from_name,
)

# TODO:
# %  DH.CREATESPIKE
# %  DH.ENUMSPIKE
# %  DH.READSPIKE
# %  DH.WRITESPIKE
# %  DH.READSPIKEINDEX
# %  DH.WRITESPIKEINDEX
# %  DH.ISCLUSTERINFO_PRESENT
# %  DH.READSPIKECLUSTER
# %  DH.WRITESPIKECLUSTER
# %  DH.GETSPIKESIZE
# %  DH.GETNUMBERSPIKES
# %  DH.GETSPIKESAMPLEPERIOD
# %  DH.GETSPIKEPARAMS
# %  DH.GETSPIKECHANDESC (-)
# %  DH.SETSPIKECHANDESC (-)


# create
@ensure_h5py_file
def create_empty_spike_group_in_file(
    file: h5py.File,
    spike_group_id: int | None,
    nSpikes: int,
    nChannels: int,
    spikeParams: SpikeParams,
    sample_period_ns: np.int32,
    n_index_items: int = 1,
    # numpy array with dtype=np.float64 of length nChannels describing calibration
    calibration: CalibrationType | None = None,
    # numpy array with dtype=CHANNELS_DTYPE of length nChannels describing channels
    channels: np.ndarray | None = None,
    name: str | None = None,
    comment: str | None = None,
) -> h5py.Group:
    # TODO: implement this function
    pass


def create_spike_group_with_data():
    pass


def enumerate_spike_groups(file: h5py.File) -> list[int]:
    return []


@ensure_h5py_file
def get_spike_group_names_from_file(
    filename: h5py.File,
) -> list[str]:
    return [
        name
        for name in filename.keys()
        if name.startswith(SPIKE_PREFIX) and isinstance(filename[name], h5py.Group)
    ]


def get_spike_groups_from_file(file: h5py.File) -> list[str]:
    """Get all spike group names from the file."""
    spike_groups = [file[name] for name in get_spike_group_names_from_file(file)]
    return spike_groups


def get_spike_group_by_id_from_file(file: h5py.File, id: int) -> h5py.Group | None:
    """Get a spike group by id from the file."""
    name = spike_name_from_id(id)
    if name in file:
        return file[name]
    return None
