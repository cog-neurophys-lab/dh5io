import h5py
import warnings
from dh5io.errors import DH5Error, DH5Warning
import numpy

CONT_PREFIX = "CONT"

DATA_DATASET_NAME = "DATA"
INDEX_DATASET_NAME = "INDEX"


def validate_cont_group(cont_group: h5py.Group) -> None:
    """Validate a CONT group in a DAQ-HDF5 file.

    This function checks if the CONT group has the required attributes and datasets.
    """
    if not isinstance(cont_group, h5py.Group):
        raise DH5Error("Not a valid HDF5 group")

    if cont_group.attrs.get("Calibration") is None:
        warnings.warn(
            message=f"Calibration attribute is missing from CONT group {cont_group.name}",
            category=DH5Warning,
        )

    if cont_group.attrs.get("SamplePeriod") is None:
        raise DH5Error(
            f"SamplePeriod attribute is missing from CONT group {cont_group.name}"
        )

    if DATA_DATASET_NAME not in cont_group:
        raise DH5Error(f"DATA dataset is missing from CONT group {cont_group.name}")

    data = cont_group[DATA_DATASET_NAME]
    if not isinstance(data, h5py.Dataset):
        raise DH5Error(f"DATA dataset in {cont_group.name} is not a dataset")

    # size of DATA must be (nSamples, nChannels)
    if len(data.shape) != 2:
        raise DH5Error(
            f"DATA dataset in {cont_group.name} has wrong shape: {data.shape}. Must be 2D"
        )

    if INDEX_DATASET_NAME not in cont_group:
        raise DH5Error(f"INDEX dataset is missing from CONT group {cont_group.name}")

    # INDEX must be a compound dataset with fields 'time' and 'offset'
    if not isinstance(cont_group[INDEX_DATASET_NAME], h5py.Dataset) or cont_group[
        INDEX_DATASET_NAME
    ].dtype.names != (
        "time",
        "offset",
    ):
        raise DH5Error(
            f"INDEX dataset in {cont_group.name} is not a named data type with fields 'time' and 'offset'"
        )

    if "Channels" in cont_group.attrs:
        channels = cont_group.attrs.get("Channels")
        if not isinstance(channels, numpy.ndarray):
            raise DH5Error(
                f"Channels attribute in {cont_group.name} is not a numpy array"
            )
        if not channels.dtype.names == (
            "GlobalChanNumber",
            "BoardChanNo",
            "ADCBitWidth",
            "MaxVoltageRange",
            "MinVoltageRange",
            "AmplifChan0",
        ):
            raise DH5Error(
                f"Channels attribute in {cont_group.name} has wrong dtype: {channels.dtype}. Must have fields 'GlobalChanNumber', 'BoardChanNo', 'ADCBitWidth', 'MaxVoltageRange', 'MinVoltageRange', 'AmplifyChan0'"
            )
    else:
        # should be an error according to specification, but is often missing
        warnings.warn(
            message=f"Channels attribute is missing from CONT group {cont_group.name}",
            category=DH5Warning,
        )
