# -*- coding: utf-8 -*-
"""MNE-Python reader for DAQ-HDF5 (DH5) files.

Provides lazy-loading Raw objects from DH5 files without copying data.
Uses dh5io for metadata access and reads HDF5 data on demand via _read_segment_file.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Literal

try:
    import psutil as _psutil
except ImportError:  # psutil is an optional dependency
    _psutil = None  # type: ignore[assignment]

import h5py
import mne
import numpy as np
import numpy.typing as npt
from mne._fiff.utils import _mult_cal_one
from mne.io.base import BaseRaw

from dh5io.dh5file import DH5File

__all__: list[str] = [
    "read_raw_dh5",
    "read_raw_dh5_per_cont",
    "epochs_from_dh5",
    "MneRawDH5",
]


# ---------------------------------------------------------------------------
# Data class for per-CONT metadata collected at init
# ---------------------------------------------------------------------------
class _ContInfo:
    __slots__ = ("id", "n_channels", "n_samples", "sfreq", "calibration", "name")

    def __init__(
        self,
        id: int,
        n_channels: int,
        n_samples: int,
        sfreq: float,
        calibration: npt.NDArray[np.float64] | None,
        name: str,
    ) -> None:
        self.id = id
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sfreq = sfreq
        self.calibration = calibration
        self.name = name


# ---------------------------------------------------------------------------
# Timestamp-to-sample mapping
# ---------------------------------------------------------------------------
def _build_region_lookup(
    index: npt.NDArray, total_samples: int, sample_period_ns: int
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Precompute arrays for fast timestamp→sample mapping.

    Returns (region_start_ns, region_end_ns, region_offset, region_n_samples)
    all with shape (n_regions,).
    """
    n = len(index)
    region_start_ns = index["time"].copy()
    region_offset = index["offset"].copy()

    region_n_samples = np.empty(n, dtype=np.int64)
    for i in range(n - 1):
        region_n_samples[i] = region_offset[i + 1] - region_offset[i]
    region_n_samples[n - 1] = total_samples - region_offset[n - 1]

    region_end_ns = region_start_ns + region_n_samples * sample_period_ns

    return region_start_ns, region_end_ns, region_offset, region_n_samples


def _ns_to_sample_time(
    time_ns: int | np.int64,
    region_start_ns: npt.NDArray[np.int64],
    region_end_ns: npt.NDArray[np.int64],
    region_offset: npt.NDArray[np.int64],
    sample_period_ns: int,
    sfreq: float,
) -> float | None:
    """Convert an absolute nanosecond timestamp to sample-based time in seconds.

    If the timestamp falls inside a recording region, map it exactly.
    If it falls in a gap between regions, snap to the nearest region boundary.
    If it's before all regions (within 10-sample tolerance), snap to start.
    Returns None if the timestamp is unmappable.
    """
    n = len(region_start_ns)

    # Check each region (region_end_ns is exclusive: start + N * period)
    for i in range(n):
        if region_start_ns[i] <= time_ns <= region_end_ns[i]:
            sample_in_region = (time_ns - region_start_ns[i]) / sample_period_ns
            return (region_offset[i] + sample_in_region) / sfreq

    # Check gaps between regions — snap to nearest boundary
    for i in range(n - 1):
        if region_end_ns[i] < time_ns < region_start_ns[i + 1]:
            dist_to_end = time_ns - region_end_ns[i]
            dist_to_start = region_start_ns[i + 1] - time_ns
            if dist_to_end <= dist_to_start:
                # Snap to last sample of region i
                last_sample_of_i = region_offset[i + 1] - 1
                return last_sample_of_i / sfreq
            else:
                # Snap to first sample of region i+1
                return region_offset[i + 1] / sfreq

    # Before first region — allow small tolerance (10 samples)
    if time_ns < region_start_ns[0]:
        if region_start_ns[0] - time_ns < sample_period_ns * 10:
            return region_offset[0] / sfreq

    # After last region — allow small tolerance (10 samples)
    if time_ns > region_end_ns[-1]:
        if time_ns - region_end_ns[-1] < sample_period_ns * 10:
            # region_end_ns is exclusive, so the last valid sample index is
            # region_offset[-1] + n_samples_in_last_region - 1; expressed as
            # (region_end_ns[-1] - region_start_ns[-1]) // sample_period_ns
            last_sample = (
                region_offset[-1]
                + (region_end_ns[-1] - region_start_ns[-1]) // sample_period_ns
                - 1
            )
            return last_sample / sfreq

    return None


def _batch_ns_to_sample_time(
    times_ns: npt.NDArray[np.int64],
    region_start_ns: npt.NDArray[np.int64],
    region_end_ns: npt.NDArray[np.int64],
    region_offset: npt.NDArray[np.int64],
    sample_period_ns: int,
    sfreq: float,
) -> npt.NDArray[np.float64]:
    """Vectorised timestamp→sample-time conversion for arrays.

    Returns array of sample-times in seconds; NaN for unmappable timestamps.

    Uses np.searchsorted for O(T log R) performance instead of O(T*R).
    """
    times_ns = np.asarray(times_ns, dtype=np.int64)
    result = np.full(len(times_ns), np.nan, dtype=np.float64)

    # --- timestamps inside a recording region ---
    # For each timestamp find which region it could belong to:
    # the last region whose start_ns <= time_ns.
    idx = np.searchsorted(region_start_ns, times_ns, side="right") - 1
    # idx == -1 means before all regions; clip to 0 for array indexing below
    candidate = np.clip(idx, 0, len(region_start_ns) - 1)

    inside = (idx >= 0) & (times_ns <= region_end_ns[candidate])
    if inside.any():
        t = times_ns[inside]
        r = candidate[inside]
        sample_in_region = (t - region_start_ns[r]).astype(np.float64) / sample_period_ns
        result[inside] = (region_offset[r] + sample_in_region) / sfreq

    # --- timestamps in gaps between regions ---
    n_regions = len(region_start_ns)
    if n_regions > 1:
        in_gap = (~inside) & (idx >= 0) & (idx < n_regions - 1)
        if in_gap.any():
            t = times_ns[in_gap]
            i = candidate[in_gap]  # region before the gap
            dist_to_end = t - region_end_ns[i]
            dist_to_start = region_start_ns[i + 1] - t
            snap_to_end = dist_to_end <= dist_to_start
            snap_times = np.where(
                snap_to_end,
                (region_offset[i + 1] - 1).astype(np.float64) / sfreq,
                region_offset[i + 1].astype(np.float64) / sfreq,
            )
            result[in_gap] = snap_times

    # --- timestamps slightly before first region ---
    before = times_ns < region_start_ns[0]
    if before.any():
        close = before & (region_start_ns[0] - times_ns < sample_period_ns * 10)
        result[close] = region_offset[0] / sfreq

    # --- timestamps slightly after last region ---
    after = times_ns > region_end_ns[-1]
    if after.any():
        close = after & (times_ns - region_end_ns[-1] < sample_period_ns * 10)
        last_sample = (
            region_offset[-1]
            + (region_end_ns[-1] - region_start_ns[-1]) // sample_period_ns
            - 1
        )
        result[close] = last_sample / sfreq

    return result


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def read_raw_dh5(
    fname: str | pathlib.Path,
    cont_ids: list[int] | Literal["all"] | None = None,
    preload: bool = False,
    verbose: bool | str | int | None = None,
) -> MneRawDH5:
    """Read a DH5 file as an MNE Raw object.

    Parameters
    ----------
    fname : str | Path
        Path to the .dh5 file.
    cont_ids : list of int | "all" | None
        Which CONT groups to include:
        - None (default): all CONT groups whose sampling rate matches the first CONT group.
        - list of int: explicitly selected CONT group IDs (error if sampling rates differ).
        - "all": all CONT groups (error if sampling rates differ).
    preload : bool
        If True, load all data into memory at init. If False, data is read lazily.
    verbose : bool | str | int | None
        MNE verbosity level.

    Returns
    -------
    raw : RawDH5
    """
    return MneRawDH5(fname, cont_ids=cont_ids, preload=preload, verbose=verbose)


def read_raw_dh5_per_cont(
    fname: str | pathlib.Path,
    preload: bool = False,
    verbose: bool | str | int | None = None,
) -> dict[int, MneRawDH5]:
    """Read a DH5 file, returning one Raw object per CONT group.

    Parameters
    ----------
    fname : str | Path
        Path to the .dh5 file.
    preload : bool
        If True, load all data into memory.
    verbose : bool | str | int | None
        MNE verbosity level.

    Returns
    -------
    raws : dict[int, RawDH5]
        Mapping from CONT group ID to Raw object.
    """
    fname = pathlib.Path(fname)
    dh5 = DH5File(fname)
    cont_ids: list[int] = dh5.get_cont_group_ids()
    del dh5
    return {
        cid: MneRawDH5(fname, cont_ids=[cid], preload=preload, verbose=verbose)
        for cid in cont_ids
    }


def epochs_from_dh5(
    raw: MneRawDH5,
    tmin: float = 0.0,
    tmax: float | None = None,
    baseline: tuple[float | None, float | None] | None = None,
) -> mne.Epochs:
    """Create MNE Epochs from a RawDH5 object using its TRIALMAP annotations.

    Parameters
    ----------
    raw : RawDH5
        A Raw object created by read_raw_dh5.
    tmin : float
        Start time relative to trial onset (seconds). Default 0.
    tmax : float | None
        End time relative to trial onset (seconds). If None, uses max trial duration.
    baseline : tuple | None
        Baseline correction window.

    Returns
    -------
    epochs : mne.Epochs
    """
    events, event_id = mne.events_from_annotations(raw, regexp="trial/.*")

    if tmax is None:
        trial_durations: list[float] = [
            float(a["duration"])
            for a in raw.annotations
            if str(a["description"]).startswith("trial/")
        ]
        if trial_durations:
            tmax = max(trial_durations)
        else:
            raise ValueError("No trial annotations found in Raw object")

    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
    )


# ---------------------------------------------------------------------------
# RawDH5 class
# ---------------------------------------------------------------------------
class MneRawDH5(BaseRaw):
    """Raw object for reading DH5 files with lazy loading.

    Parameters
    ----------
    fname : str | Path
        Path to the .dh5 file.
    cont_ids : list of int | "all" | None
        Which CONT groups to include. See read_raw_dh5 for details.
    preload : bool
        If True, load all data into memory.
    verbose : bool | str | int | None
        MNE verbosity level.
    """

    def __init__(
        self,
        fname: str | pathlib.Path,
        cont_ids: list[int] | Literal["all"] | None = None,
        preload: bool = False,
        verbose: bool | str | int | None = None,
    ) -> None:
        fname = pathlib.Path(fname)

        # Collect metadata using dh5io
        dh5 = DH5File(fname)
        all_conts: list[_ContInfo] = []
        for cont in dh5.get_cont_groups():
            sfreq = 1e9 / cont.sample_period
            calib = cont.calibration
            if calib is not None:
                calib = np.array(calib, dtype=np.float64)
            name: str = cont.name if cont.name else f"CONT{cont.id}"
            all_conts.append(
                _ContInfo(
                    id=cont.id,
                    n_channels=cont.n_channels,
                    n_samples=cont.n_samples,
                    sfreq=sfreq,
                    calibration=calib,
                    name=name,
                )
            )

        # Select CONT groups
        selected = self._select_conts(all_conts, cont_ids, fname)
        sfreq = selected[0].sfreq

        # Handle differing sample counts
        n_samples_set = set(c.n_samples for c in selected)
        if len(n_samples_set) > 1:
            total_samples: int = min(n_samples_set)
            warnings.warn(
                f"CONT groups have different sample counts: {n_samples_set}. "
                f"Using minimum: {total_samples}."
            )
        else:
            total_samples = selected[0].n_samples

        # Build channel map: (cont_id, column_index, calibration_value, ch_name)
        channel_map: list[tuple[int, int, float, str]] = []
        ch_names: list[str] = []
        for c in selected:
            for col in range(c.n_channels):
                cal: float = (
                    float(c.calibration[col]) if c.calibration is not None else 1.0
                )
                ch_name: str = f"{c.name}/{col}"
                channel_map.append((c.id, col, cal, ch_name))
                ch_names.append(ch_name)

        # Changed from "misc" to "ecog" so MNE recognizes these as data channels
        info: mne.Info = mne.create_info(ch_names, sfreq, ch_types="ecog")

        # Set per-channel calibration so MNE's cals mechanism applies it
        with info._unlock():
            for i, (_, _, cal, _) in enumerate(channel_map):
                info["chs"][i]["cal"] = cal

        raw_extras: dict = {
            "channel_map": channel_map,
            "total_samples": total_samples,
        }

        # Warn if preloading would exceed half of available memory
        if preload:
            n_channels = len(channel_map)
            data_bytes = n_channels * total_samples * 8  # float64 = 8 bytes
            if _psutil is not None:
                avail = _psutil.virtual_memory().available
                if data_bytes > avail // 2:
                    warnings.warn(
                        f"preload=True will load ~{data_bytes / 2**30:.1f} GiB into memory, "
                        f"but only {avail / 2**30:.1f} GiB is available. "
                        "Consider using preload=False for lazy loading.",
                        ResourceWarning,
                        stacklevel=3,
                    )

        super().__init__(
            info,
            preload=preload,
            filenames=[str(fname)],
            raw_extras=[raw_extras],
            last_samps=[total_samples - 1],
            orig_format="short",
            verbose=verbose,
        )

        # Build region lookup from first selected CONT's INDEX
        sample_period_ns: int = int(1e9 / sfreq)
        first_cont = dh5.get_cont_group_by_id(selected[0].id)
        index: npt.NDArray = first_cont.index
        self._region_lookup = _build_region_lookup(
            index, total_samples, sample_period_ns
        )
        self._sample_period_ns: int = sample_period_ns
        self._sfreq: float = sfreq

        # Add annotations
        self._add_dh5_annotations(dh5)
        del dh5

    @staticmethod
    def _select_conts(
        all_conts: list[_ContInfo],
        cont_ids: list[int] | Literal["all"] | None,
        fname: pathlib.Path,
    ) -> list[_ContInfo]:
        """Select CONT groups based on cont_ids parameter."""
        if not all_conts:
            raise ValueError(f"No CONT groups found in {fname}")

        if cont_ids is None:
            target_sfreq: float = all_conts[0].sfreq
            selected = [c for c in all_conts if c.sfreq == target_sfreq]
            skipped = [c for c in all_conts if c.sfreq != target_sfreq]
            if skipped:
                names = [f"CONT{c.id} ({c.sfreq:.1f} Hz)" for c in skipped]
                warnings.warn(
                    f"Skipping CONT groups with non-matching sampling rate: "
                    f"{', '.join(names)}. Target rate: {target_sfreq:.1f} Hz."
                )
        elif cont_ids == "all":
            selected = all_conts
            sfreqs = set(c.sfreq for c in selected)
            if len(sfreqs) > 1:
                raise ValueError(
                    f"cont_ids='all' but CONT groups have different sampling rates: "
                    f"{sfreqs}. Use cont_ids=None to auto-select matching rates."
                )
        else:
            id_set = set(cont_ids)
            selected = [c for c in all_conts if c.id in id_set]
            missing = id_set - {c.id for c in selected}
            if missing:
                raise ValueError(f"CONT group IDs not found: {missing}")
            sfreqs = set(c.sfreq for c in selected)
            if len(sfreqs) > 1:
                raise ValueError(
                    f"Selected CONT groups have different sampling rates: {sfreqs}"
                )

        if not selected:
            raise ValueError(f"No CONT groups selected from {fname}")
        return selected

    def _add_dh5_annotations(self, dh5: DH5File) -> None:
        """Add TRIALMAP and EV02 as MNE Annotations."""
        region_start_ns, region_end_ns, region_offset, _ = self._region_lookup

        onsets: list[float] = []
        durations: list[float] = []
        descriptions: list[str] = []

        # Region boundaries as BAD annotations
        for i in range(len(region_start_ns) - 1):
            if region_start_ns[i + 1] > region_end_ns[i]:
                boundary_sample: float = float(region_offset[i + 1]) / self._sfreq
                onsets.append(boundary_sample)
                durations.append(0.0)
                descriptions.append("BAD_region_boundary")

        # TRIALMAP
        trialmap = dh5.get_trialmap()
        if trialmap is not None and len(trialmap) > 0:
            start_times = _batch_ns_to_sample_time(
                trialmap.start_time_nanoseconds,
                region_start_ns,
                region_end_ns,
                region_offset,
                self._sample_period_ns,
                self._sfreq,
            )
            end_times = _batch_ns_to_sample_time(
                trialmap.end_time_nanoseconds,
                region_start_ns,
                region_end_ns,
                region_offset,
                self._sample_period_ns,
                self._sfreq,
            )
            for i in range(len(trialmap)):
                if np.isnan(start_times[i]) or np.isnan(end_times[i]):
                    continue
                dur: float = end_times[i] - start_times[i]
                if dur > 0:
                    onsets.append(start_times[i])
                    durations.append(dur)
                    stim_no: int = int(trialmap.trial_type_numbers[i])
                    outcome: int = int(trialmap.trial_outcomes_integer[i])
                    descriptions.append(f"trial/StimNo={stim_no}/Outcome={outcome}")

        # EV02 events
        events_arr = dh5.get_events_array()
        if events_arr is not None and len(events_arr) > 0:
            event_times = _batch_ns_to_sample_time(
                events_arr["time"],
                region_start_ns,
                region_end_ns,
                region_offset,
                self._sample_period_ns,
                self._sfreq,
            )
            for j in range(len(events_arr)):
                if np.isnan(event_times[j]):
                    continue
                onsets.append(event_times[j])
                durations.append(0.0)
                descriptions.append(f"event/{events_arr['event'][j]}")

        if onsets:
            annot = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
            )
            self.set_annotations(annot, emit_warning=False)

    def _read_segment_file(
        self,
        data: npt.NDArray[np.float64],
        idx: npt.NDArray[np.intp] | slice,
        fi: int,
        start: int,
        stop: int,
        cals: npt.NDArray[np.float64] | None,
        mult: npt.NDArray[np.float64] | None,
    ) -> None:
        """Read a segment of data from the DH5 file."""
        channel_map: list[tuple[int, int, float, str]] = self._raw_extras[fi][
            "channel_map"
        ]

        with h5py.File(self.filenames[fi], "r") as f:
            # Group channels by CONT ID for efficient HDF5 reads
            cont_reads: dict[int, list[tuple[int, int]]] = {}
            for mne_ch_idx, (cont_id, col, _, _) in enumerate(channel_map):
                if cont_id not in cont_reads:
                    cont_reads[cont_id] = []
                cont_reads[cont_id].append((mne_ch_idx, col))

            n_channels: int = len(channel_map)
            n_times: int = stop - start
            one: npt.NDArray[np.float64] = np.empty(
                (n_channels, n_times), dtype=np.float64
            )

            for cont_id, ch_list in cont_reads.items():
                ds = f[f"CONT{cont_id}/DATA"]
                cols = [col for _, col in ch_list]

                if len(cols) == 1:
                    # Single channel: direct read
                    one[ch_list[0][0]] = ds[start:stop, cols[0]].astype(np.float64)
                else:
                    # Multiple channels: read full slice
                    block: npt.NDArray[np.float64] = ds[start:stop, :].T.astype(
                        np.float64
                    )
                    for mne_idx, col in ch_list:
                        one[mne_idx] = block[col]

        _mult_cal_one(data, one, idx, cals, mult)
