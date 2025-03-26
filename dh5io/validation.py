import pathlib
import numpy
import h5py
from dh5io.errors import DH5Error, DH5Warning
from dh5io.dh5file import get_cont_groups_from_file
import warnings


def validate_dh5_file(filename: str | pathlib.Path | h5py.File) -> None:
    """Validate if the given file is a valid DAQ-HDF5 file.

    This function checks if the file has the required attributes and groups.
    """

    if isinstance(filename, (str, pathlib.Path)):
        file = h5py.File(filename, "r")

    if not isinstance(filename, (str, pathlib.Path, h5py.File)):
        raise TypeError("filename must be a str, pathlib.Path or h5py.File")

    if not isinstance(file, h5py.File):
        raise DH5Error("Not a valid HDF5 file")

    if file.attrs.get("FILEVERSION") is None:
        raise DH5Error("FILEVERSION attribute is missing")

    # check for named data type CONT_INDEX_ITEM
    if "CONT_INDEX_ITEM" not in file:
        raise DH5Error("CONT_INDEX_ITEM not found")

    # CONT_INDEX_ITEM must be a compound data type with time and offset
    cont_dtype: h5py.Datatype = file["CONT_INDEX_ITEM"]
    if not isinstance(cont_dtype, h5py.Datatype) or cont_dtype.dtype.names != (
        "time",
        "offset",
    ):
        raise DH5Error(
            "CONT_INDEX_ITEM is not a named data type with fields 'time' and 'offset'"
        )

    # check for CONT groups
    cont_groups = get_cont_groups_from_file(file)
    for cont_group in cont_groups:
        validate_cont_group(cont_group)

    # check for TRIALMAP dataset
    if "TRIALMAP" not in file:
        warnings.warn(
            message=f"TRIALMAP dataset is missing from {filename}", category=DH5Warning
        )


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

    if "DATA" not in cont_group:
        raise DH5Error(f"DATA dataset is missing from CONT group {cont_group.name}")

    data = cont_group["DATA"]
    if not isinstance(data, h5py.Dataset):
        raise DH5Error(f"DATA dataset in {cont_group.name} is not a dataset")

    # size of DATA must be (nSamples, nChannels)
    if len(data.shape) != 2:
        raise DH5Error(
            f"DATA dataset in {cont_group.name} has wrong shape: {data.shape}. Must be 2D"
        )

    if "INDEX" not in cont_group:
        raise DH5Error(f"INDEX dataset is missing from CONT group {cont_group.name}")

    # INDEX must be a compound dataset with fields 'time' and 'offset'
    if not isinstance(cont_group["INDEX"], h5py.Dataset) or cont_group[
        "INDEX"
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


def validate_trialmap_dataset(trialmap: h5py.Dataset) -> None:
    # trialmap must be a compound dataset with fields 'TrialNo', 'StimNo', 'Outcome', 'StartTime', 'EndTime'
    if not isinstance(trialmap, h5py.Dataset) or trialmap.dtype.names != (
        "TrialNo",
        "StimNo",
        "Outcome",
        "StartTime",
        "EndTime",
    ):
        raise DH5Error(
            f"TRIALMAP dataset is not a named data type with fields 'TrialNo', 'StimNo', 'Outcome', 'StartTime', 'EndTime': {trialmap.dtype}"
        )
