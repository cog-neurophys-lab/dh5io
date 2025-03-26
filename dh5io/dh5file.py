import pathlib
from unicodedata import category
import numpy
import numpy.typing as npt
import typing
import h5py
import warnings

_trialmap_dtype = numpy.dtype(
    [
        ("TrialNo", "int32"),
        ("StimNo", "int32"),
        ("Outcome", "int32"),
        ("StartTime", "int64"),
        ("EndTime", "int64"),
    ]
)


def get_cont_group_names_from_file(
    filename: str | pathlib.Path | h5py.File,
) -> list[str]:
    if isinstance(filename, (str, pathlib.Path)):
        with h5py.File(filename, "r") as file:
            return [
                name
                for name in file.keys()
                if name.startswith("CONT") and isinstance(file[name], h5py.Group)
            ]

    if isinstance(filename, h5py.File):
        return [
            name
            for name in filename.keys()
            if name.startswith("CONT") and isinstance(filename[name], h5py.Group)
        ]

    raise TypeError("filename must be a str, pathlib.Path or h5py.File")


def get_cont_groups_from_file(file: str | pathlib.Path | h5py.File) -> list[h5py.Group]:
    cont_group_names = get_cont_group_names_from_file(file)
    if isinstance(file, (str, pathlib.Path)):
        with h5py.File(file, "r") as file:
            return [file[name] for name in cont_group_names]
    if isinstance(file, h5py.File):
        return [file[name] for name in cont_group_names]

    raise TypeError("file must be a str, pathlib.Path or h5py.File")


class DH5File:
    """Class for interacting with DAQ-HDF5 (*.dh5) files from the Kreiter lab.

    The file format ist based on HDF5. See https://github.com/cog-neurophys-lab/DAQ-HDF5 for
    the specification of the format.
    """

    file: h5py.File

    def __init__(self, filename: str | pathlib.Path, mode="r"):
        self.file = h5py.File(filename, mode)

    def __del__(self):
        self.file.close()

    def __str__(self):
        return f"""
        DAQ-HDF5 File (version {self.version}) {self.file.filename:s} containing:
            ├─── {len(self.get_cont_groups()):5d} CONT Groups: {self.get_cont_group_names()}
            ├─── {len(self.get_spike_groups()):5d} SPIKE Groups: {self.get_spike_group_names()}
            ├─── {len(self.get_events()):5d} Events
            └─── {len(self.get_trialmap()):5d} Trials in TRIALMAP
        """

    @property
    def version(self) -> int | None:
        return self.file.attrs.get("FILEVERSION")

    # cont groups
    def get_cont_groups(self) -> list[h5py.Group]:
        return get_cont_groups_from_file(self.file)

    def get_cont_group_names(self) -> list[str]:
        return get_cont_groups_from_file(self.file)

    def get_cont_group_ids(self) -> list[int]:
        return [int(name.split("CONT")[1]) for name in self.get_cont_group_names()]

    def get_cont_group_by_id(self, id: int) -> h5py.Group:
        contGroup = self.file.get(f"CONT{id}")
        if contGroup is None:
            raise DH5Error(f"CONT{id} does not exist in {self.file.filename}")
        return contGroup

    def get_cont_data_by_id(self, cont_id: int) -> numpy.ndarray:
        return numpy.array(self.get_cont_group_by_id(cont_id).get("DATA")[:])

    def get_calibrated_cont_data_by_id(self, cont_id: int) -> numpy.ndarray:
        calibration = self.get_cont_group_by_id(cont_id).attrs.get("Calibration")
        if calibration is None:
            return self.get_cont_data_by_id(cont_id)
        return self.get_cont_data_by_id(cont_id) * calibration

    def get_cont_size(self, cont_id) -> tuple[int, int]:
        nSamples, nChannels = self.get_cont_data_by_id(cont_id).shape
        return (nSamples, nChannels)

    # spike groups
    def get_spike_groups(self) -> list[h5py.Group]:
        return [self.file[name] for name in self.get_spike_group_names()]

    def get_spike_group_names(self) -> list[str]:
        return [
            name
            for name in self.file.keys()
            if name.startswith("SPIKE") and isinstance(self.file[name], h5py.Group)
        ]

    def get_spike_group_by_id(self, id: int) -> h5py.Group | None:
        return self.file.get(f"SPIKE{id}")

    def get_cont_index_by_id(self, cont_id: int) -> h5py.Dataset:
        return self.get_cont_group_by_id(cont_id).get("INDEX")

    # trialmap
    def get_trialmap(self) -> numpy.ndarray | None:
        return numpy.array(self.file.get("TRIALMAP"), dtype=_trialmap_dtype)

    def get_events(self) -> h5py.Dataset | None:
        return self.file.get("EV02")

    @staticmethod
    def get_cont_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("CONT"))

    @staticmethod
    def get_spike_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("SPIKE"))


def validate_dh5_file(filename: str | pathlib.Path) -> None:
    """Validate if the given file is a valid DAQ-HDF5 file.

    This function checks if the file has the required attributes and groups.
    """

    file = h5py.File(filename, "r")

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


def create_dh5_file(filename: str | pathlib.Path, CONT_INDEX_ITEM=None):
    pass


class DH5Error(Exception):
    pass


class DH5Warning(Warning):
    pass
