import pathlib
import numpy
import h5py
from dh5io.errors import DH5Error, DH5Warning

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
        return get_cont_group_names_from_file(self.file)

    def get_cont_group_ids(self) -> list[int]:
        cont_group_names = self.get_cont_group_names()
        return [int(name.split("CONT")[1]) for name in cont_group_names]

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


def create_dh5_file(filename: str | pathlib.Path, CONT_INDEX_ITEM=None):
    pass
