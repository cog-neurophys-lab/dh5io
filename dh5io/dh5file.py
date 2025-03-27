"""
DAQ-HDF (DH5) is a set of specifications on how to store electrophysiological
data in files based on the HDF5 file format.

The following kinds of data can be stored in DAQ-HDF files:

-   signal data (CONT groups)
-   spike data (SPIKE groups)
-   trialmap (TRIALMAP dataset)
-   time markers (Markers group) and intervals (Intervals group)
-   processing history (Operations groups)

There are additional two kinds of data which are specified to
accommodate the respective streams from DAQ-files and other similar file
formats such as UFF:

-   event triggers (EV02 dataset)
-   trial descriptor records (TD01 dataset)

These sets of data are only used for subsequent generation of trialmap,
time markers and intervals based on the information from them.

DAQ-HDF has the following **attributes** associated with the root group:

- `FILEVERSION` (`int32` scalar) – version of the DAQ-HDF. The current version number is 2,
and this is the only version described in this document. If this attribute is missing,
version 1 is assumed. Version 1 is obsolete, and it has substantial differences in data
structures compared to version 2.
- `BOARDS` (`string` array) – names of the A/D boards used during recording of data. If
initial data was acquired by means other than analog recording, for example, if it was
generated in software, this attribute may contain some description of the creation process
instead.

The root group must also contain a shared *datatype* named `CONT_INDEX_ITEM`
if there are `CONT` blocks present in the file. See description of this
datatype in the `CONT` blocks description.

"""

import pathlib
import numpy
import h5py
import dh5io.trialmap as trialmap
import dh5io.event_triggers as event_triggers
import dh5io.cont as cont

from dh5io.errors import DH5Error, DH5Warning


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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
        return cont.get_cont_groups_from_file(self.file)

    def get_cont_group_names(self) -> list[str]:
        return cont.get_cont_group_names_from_file(self.file)

    def get_cont_group_ids(self) -> list[int]:
        return cont.enumerate_cont_groups(self.file)

    def get_cont_group_by_id(self, id: int) -> h5py.Group:
        return cont.get_cont_group_by_id_from_file(self.file, id)

    def get_cont_data_by_id(self, cont_id: int) -> numpy.ndarray:
        return cont.get_cont_data_by_id_from_file(self.file, cont_id)

    def get_calibrated_cont_data_by_id(self, cont_id: int) -> numpy.ndarray:
        return cont.get_calibrated_cont_data_by_id(self.file, cont_id)

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
        return trialmap.get_trialmap_from_file(self.file)

    def get_events(self) -> h5py.Dataset | None:
        return event_triggers.get_event_triggers_from_file(self.file)

    @staticmethod
    def get_spike_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("SPIKE"))
