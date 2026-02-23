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


def is_continuous(fname: str | pathlib.Path) -> bool:
    """Return True if every CONT block in the file contains exactly one region.

    Parameters
    ----------
    fname : str | Path
        Path to the DH5 file.

    Returns
    -------
    bool
        True when all CONT blocks have a single region (no acquisition gaps).
    """
    with DH5File(fname) as dh5:
        return dh5.is_continuous()


def cont_blocks_start_simultaneously(fname: str | pathlib.Path) -> bool:
    """Return True if all CONT blocks in the file share the same first-region start timestamp.

    Parameters
    ----------
    fname : str | Path
        Path to the DH5 file.

    Returns
    -------
    bool
        True when every CONT block's INDEX[0]["time"] is identical, or when the
        file contains fewer than two CONT blocks.
    """
    with DH5File(fname) as dh5:
        return dh5.cont_blocks_start_simultaneously()


class DH5File:
    """Class for interacting with DAQ-HDF5 (*.dh5) files from the Kreiter lab.

    The file format ist based on HDF5. See https://github.com/cog-neurophys-lab/DAQ-HDF5 for
    the specification of the format.
    """

    _file: h5py.File

    def __init__(self, filename: str | pathlib.Path, mode="r"):
        if not pathlib.Path(filename).exists():
            raise FileNotFoundError(
                f"File {filename} does not exist. To create a new valid DH5 file use `dh5io.create_dh5_file`"
            )
        self._file = h5py.File(filename, mode)

    def __del__(self):
        if hasattr(self, "_file") and self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file:
            self._file.close()

    def __repr__(self) -> str:
        return f"DH5File({self._file.filename!r}, mode={self._file.mode!r})"

    def __str__(self):
        cont_by_sfreq = self.get_cont_groups_by_sfreq()
        n_cont = sum(len(g) for g in cont_by_sfreq.values())
        continuous = self.is_continuous()
        simultaneous = self.cont_blocks_start_simultaneously()
        cont_flags = []
        if continuous:
            cont_flags.append("continuous")
        else:
            cont_flags.append("discontinuous")
        if n_cont > 1:
            cont_flags.append(
                "simultaneous start" if simultaneous else "non-simultaneous start"
            )
        cont_flags_str = f"  [{', '.join(cont_flags)}]" if cont_flags else ""

        cont_groups_lines = []
        sfreqs = list(cont_by_sfreq.keys())
        for si, sfreq in enumerate(sfreqs):
            branch = "├" if si < len(sfreqs) - 1 else "└"
            cont_groups_lines.append(f"        │   {branch}─── {sfreq:g} Hz")
            conts = cont_by_sfreq[sfreq]
            inner = "│" if si < len(sfreqs) - 1 else " "
            for ni, c in enumerate(conts):
                nbranch = "├" if ni < len(conts) - 1 else "└"
                info = f"{c.n_channels}ch, {c.n_samples} samples"
                if c.n_regions > 1:
                    info += f", {c.n_regions} regions"
                label = (
                    f"CONT{c.id}: {c.name} — {info}"
                    if c.name
                    else f"CONT{c.id} — {info}"
                )
                cont_groups_lines.append(f"        │   {inner}   {nbranch}─── {label}")
        cont_groups_str = (
            "\n".join(cont_groups_lines)
            if cont_groups_lines
            else "        │   └── (none)"
        )

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
        ├───CONT Groups ({n_cont:d}){cont_flags_str}:
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
        val = self._file.attrs.get(BOARDS_ATTRIBUTE_NAME)
        if val is not None:
            return [b.decode() if isinstance(b, bytes) else b for b in val]
        return val

    # cont groups
    def get_cont_groups(self) -> list[cont.Cont]:
        return [
            cont.Cont(group) for group in cont.get_cont_groups_from_file(self._file)
        ]

    def get_cont_groups_by_sfreq(self) -> dict[float, list[cont.Cont]]:
        """Return CONT groups grouped by sampling rate (Hz)."""
        groups: dict[float, list[cont.Cont]] = {}
        for c in self.get_cont_groups():
            sfreq = 1e9 / c.sample_period
            groups.setdefault(sfreq, []).append(c)
        return groups

    def get_cont_groups_by_ids(self, ids: list[int]) -> list[cont.Cont]:
        """Return CONT groups for the given IDs, preserving the requested order.

        Raises
        ------
        DH5Error
            If any of the requested IDs are not present in the file.
        """
        from dh5io.errors import DH5Error

        all_conts = {c.id: c for c in self.get_cont_groups()}
        missing = set(ids) - all_conts.keys()
        if missing:
            raise DH5Error(
                f"CONT group IDs not found in {self._file.filename}: {missing}"
            )
        return [all_conts[i] for i in ids]

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
    # TODO:
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

    def is_continuous(self) -> bool:
        """Return True if every CONT block contains exactly one region.

        A file is considered continuous when no CONT block has gaps in its
        recording (i.e. all INDEX arrays have length 1).

        Returns
        -------
        bool
            True when all CONT blocks have a single region, or when the file
            contains no CONT blocks.
        """
        return all(c.n_regions == 1 for c in self.get_cont_groups())

    def cont_blocks_start_simultaneously(self) -> bool:
        """Return True if all CONT blocks share the same first-region start timestamp.

        Returns
        -------
        bool
            True when every CONT block's INDEX[0]["time"] is identical, or when
            the file contains fewer than two CONT blocks.
        """
        conts = self.get_cont_groups()
        if len(conts) < 2:
            return True
        first_time = conts[0].index[0]["time"]
        return all(c.index[0]["time"] == first_time for c in conts[1:])

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
        data = trialmap.get_trialmap_from_file(self._file)
        return trialmap.Trialmap(data) if data is not None else None

    def get_events_dataset(self) -> h5py.Dataset | None:
        return event_triggers.get_event_triggers_dataset_from_file(self._file)

    def get_events_array(self) -> numpy.ndarray | None:
        return event_triggers.get_event_triggers_from_file(self._file)

    @staticmethod
    def get_spike_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("SPIKE"))
