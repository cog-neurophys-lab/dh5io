"""
DAQ-HDF (DH5) is a set of specifications on how to store electrophysiological
data in files based on the HDF5 file format.

The following kinds of data can be stored in DAQ-HDF files:

-   signal data (CONT groups)
-   spike data (SPIKE groups)
-   wavelet data (WAVELET groups)
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

import h5py
import numpy

import dh5io.cont as cont
import dh5io.event_triggers as event_triggers
import dh5io.trialmap as trialmap
import dh5io.wavelet as wavelet
from dhspec.dh5file import BOARDS_ATTRIBUTE_NAME, FILEVERSION_ATTRIBUTE_NAME


def dh5file_from_h5file(file: h5py.File):
    return DH5File(file.filename, mode=file.mode)


class DH5File:
    """Class for interacting with DAQ-HDF5 (*.dh5) files from the Kreiter lab.

    The file format ist based on HDF5. See https://github.com/cog-neurophys-lab/DAQ-HDF5 for
    the specification of the format.
    """

    _file: h5py.File

    def __init__(self, filename: str | pathlib.Path, mode="r"):
        if not pathlib.Path(filename).exists():
            raise FileNotFoundError(
                f"File {filename} does not exist. To create a new valid DH5 file use `dh5io.create.create_dh5_file`"
            )
        self._file = h5py.File(filename, mode)

    def __del__(self):
        if self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file:
            self._file.close()

    def __str__(self):
        cont_group_names = self.get_cont_group_names()
        cont_groups_str = ""
        if cont_group_names:
            cont_groups_lines = [
                f"        │   ├─── {name}" for name in cont_group_names[:-1]
            ]
            cont_groups_lines.append(f"        │   └─── {cont_group_names[-1]}")
            cont_groups_str = "\n".join(cont_groups_lines)
        else:
            cont_groups_str = "        │   └── (none)"

        spike_group_names = self.get_spike_group_names()
        spike_groups_str = ""
        if spike_group_names:
            spike_groups_lines = [
                f"        │   ├─── {name}" for name in spike_group_names[:-1]
            ]
            spike_groups_lines.append(f"        │   └─── {spike_group_names[-1]}")
            spike_groups_str = "\n".join(spike_groups_lines)
        else:
            spike_groups_str = "        │   └── (none)"

        wavelet_group_names = self.get_wavelet_group_names()
        wavelet_groups_str = ""
        if wavelet_group_names:
            wavelet_groups_lines = [
                f"        │   ├─── {name}" for name in wavelet_group_names[:-1]
            ]
            wavelet_groups_lines.append(f"        │   └─── {wavelet_group_names[-1]}")
            wavelet_groups_str = "\n".join(wavelet_groups_lines)
        else:
            wavelet_groups_str = "        │   └── (none)"

        events_dataset = self.get_events_dataset()
        n_events = len(events_dataset) if events_dataset is not None else 0
        try:
            trialmap = self.get_trialmap()
            n_trials = len(trialmap) if trialmap is not None else 0
        except (AttributeError, Exception):
            n_trials = 0

        return f"""
    DAQ-HDF5 File (version {self.version}) {self._file.filename:s} containing:
        ├───CONT Groups ({len(cont_group_names):d}):
{cont_groups_str}
        ├───SPIKE Groups ({len(spike_group_names):d}):
{spike_groups_str}
        ├───WAVELET Groups ({len(wavelet_group_names):d}):
{wavelet_groups_str}
        ├─── {n_events:d} Events
        └─── {n_trials:d} Trials in TRIALMAP
        """

    @property
    def version(self) -> int | None:
        return self._file.attrs.get(FILEVERSION_ATTRIBUTE_NAME)

    @property
    def boards(self) -> list[str] | None:
        return self._file.attrs.get(BOARDS_ATTRIBUTE_NAME)

    # cont groups
    def get_cont_groups(self) -> list[cont.Cont]:
        return [
            cont.Cont(group) for group in cont.get_cont_groups_from_file(self._file)
        ]

    def get_cont_group_names(self) -> list[str]:
        return cont.get_cont_group_names_from_file(self._file)

    def get_cont_group_ids(self) -> list[int]:
        return cont.enumerate_cont_groups(self._file)

    def get_cont_group_by_id(self, id: int) -> cont.Cont:
        return cont.Cont(cont.get_cont_group_by_id_from_file(self._file, id))

    def get_cont_data_by_id(self, cont_id: int) -> numpy.ndarray:
        return cont.get_cont_data_by_id_from_file(self._file, cont_id)

    def get_calibrated_cont_data_by_id(self, cont_id: int) -> numpy.ndarray:
        return cont.get_calibrated_cont_data_by_id(self._file, cont_id)

    def get_cont_size(self, cont_id) -> tuple[int, int]:
        nSamples, nChannels = self.get_cont_data_by_id(cont_id).shape
        return (nSamples, nChannels)

    # spike groups
    def get_spike_groups(self) -> list[h5py.Group]:
        return [self._file[name] for name in self.get_spike_group_names()]

    def get_spike_group_names(self) -> list[str]:
        return [
            name
            for name in self._file.keys()
            if name.startswith("SPIKE") and isinstance(self._file[name], h5py.Group)
        ]

    def get_spike_group_by_id(self, id: int) -> h5py.Group | None:
        return self._file.get(f"SPIKE{id}")

    def get_cont_index_by_id(self, cont_id: int) -> h5py.Dataset:
        return self.get_cont_group_by_id(cont_id).get("INDEX")

    # wavelet groups
    def get_wavelet_groups(self) -> list[wavelet.Wavelet]:
        return [
            wavelet.Wavelet(group)
            for group in wavelet.get_wavelet_groups_from_file(self._file)
        ]

    def get_wavelet_group_names(self) -> list[str]:
        return wavelet.get_wavelet_group_names_from_file(self._file)

    def get_wavelet_group_ids(self) -> list[int]:
        return wavelet.enumerate_wavelet_groups(self._file)

    def get_wavelet_group_by_id(self, id: int) -> wavelet.Wavelet | None:
        group = wavelet.get_wavelet_group_by_id_from_file(self._file, id)
        return wavelet.Wavelet(group) if group is not None else None

    # trialmap
    def get_trialmap(self) -> trialmap.Trialmap | None:
        return trialmap.Trialmap(trialmap.get_trialmap_from_file(self._file))

    def get_events_dataset(self) -> h5py.Dataset | None:
        return event_triggers.get_event_triggers_dataset_from_file(self._file)

    def get_events_array(self) -> numpy.ndarray | None:
        return event_triggers.get_event_triggers_from_file(self._file)

    @staticmethod
    def get_spike_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("SPIKE"))
