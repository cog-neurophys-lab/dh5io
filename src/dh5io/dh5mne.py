# -*- coding: utf-8 -*-
"""MNE-Python reader for DAQ-HDF5 (DH5) files.

Provides lazy-loading Raw objects from DH5 files without copying data.
Uses dh5io for metadata access and reads HDF5 data on demand via _read_segment_file.
"""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass

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

from dh5io.cont import Cont
from dh5io.dh5file import DH5File
from dh5io.errors import DH5Error, DH5Warning, DH5DiscontinuousRegionsWarning, DH5SampleCountMismatchWarning, DH5SampleRateMismatchWarning

__all__: list[str] = [
    "read_raw_dh5",
    "read_raw_dh5_per_sfreq",
    "read_raw_dh5_per_cont",
    "epochs_from_dh5",
    "annotations_from_dh5",
    "MneRawDH5",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class _RegionLookup:
    """Precomputed arrays for fast timestamp→sample mapping.

    All arrays have shape (n_regions,). region_end_ns is exclusive
    (= region_start_ns + n_samples * sample_period_ns).
    """

    start_ns: npt.NDArray[np.int64]
    end_ns: npt.NDArray[np.int64]
    offset: npt.NDArray[np.int64]
    n_samples: npt.NDArray[np.int64]


# ---------------------------------------------------------------------------
# Timestamp-to-sample mapping
# ---------------------------------------------------------------------------
def _build_region_lookup(
    index: npt.NDArray, total_samples: int, sample_period_ns: int
) -> _RegionLookup:
    """Precompute arrays for fast timestamp→sample mapping."""
    n = len(index)
    start_ns = index["time"].copy()
    offset = index["offset"].copy()

    n_samples = np.empty(n, dtype=np.int64)
    for i in range(n - 1):
        n_samples[i] = offset[i + 1] - offset[i]
    n_samples[n - 1] = total_samples - offset[n - 1]

    end_ns = start_ns + n_samples * sample_period_ns

    return _RegionLookup(start_ns=start_ns, end_ns=end_ns, offset=offset, n_samples=n_samples)


def _ns_to_sample_time(
    time_ns: int | np.int64,
    rl: _RegionLookup,
    sample_period_ns: int,
    sfreq: float,
) -> float | None:
    """Convert an absolute nanosecond timestamp to sample-based time in seconds.

    If the timestamp falls inside a recording region, map it exactly.
    If it falls in a gap between regions, snap to the nearest region boundary.
    If it's before all regions (within 10-sample tolerance), snap to start.
    Returns None if the timestamp is unmappable.
    """
    n = len(rl.start_ns)

    # Check each region (end_ns is exclusive: start + N * period)
    for i in range(n):
        if rl.start_ns[i] <= time_ns <= rl.end_ns[i]:
            sample_in_region = (time_ns - rl.start_ns[i]) / sample_period_ns
            return (rl.offset[i] + sample_in_region) / sfreq

    # Check gaps between regions — snap to nearest boundary
    for i in range(n - 1):
        if rl.end_ns[i] < time_ns < rl.start_ns[i + 1]:
            dist_to_end = time_ns - rl.end_ns[i]
            dist_to_start = rl.start_ns[i + 1] - time_ns
            if dist_to_end <= dist_to_start:
                # Snap to last sample of region i
                return (rl.offset[i + 1] - 1) / sfreq
            else:
                # Snap to first sample of region i+1
                return rl.offset[i + 1] / sfreq

    # Before first region — allow small tolerance (10 samples)
    if time_ns < rl.start_ns[0]:
        if rl.start_ns[0] - time_ns < sample_period_ns * 10:
            return rl.offset[0] / sfreq

    # After last region — allow small tolerance (10 samples)
    if time_ns > rl.end_ns[-1]:
        if time_ns - rl.end_ns[-1] < sample_period_ns * 10:
            last_sample = (
                rl.offset[-1]
                + (rl.end_ns[-1] - rl.start_ns[-1]) // sample_period_ns
                - 1
            )
            return last_sample / sfreq

    return None


def _batch_ns_to_sample_time(
    times_ns: npt.NDArray[np.int64],
    rl: _RegionLookup,
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
    idx = np.searchsorted(rl.start_ns, times_ns, side="right") - 1
    # idx == -1 means before all regions; clip to 0 for array indexing below
    candidate = np.clip(idx, 0, len(rl.start_ns) - 1)

    inside = (idx >= 0) & (times_ns <= rl.end_ns[candidate])
    if inside.any():
        t = times_ns[inside]
        r = candidate[inside]
        sample_in_region = (t - rl.start_ns[r]).astype(np.float64) / sample_period_ns
        result[inside] = (rl.offset[r] + sample_in_region) / sfreq

    # --- timestamps in gaps between regions ---
    n_regions = len(rl.start_ns)
    if n_regions > 1:
        in_gap = (~inside) & (idx >= 0) & (idx < n_regions - 1)
        if in_gap.any():
            t = times_ns[in_gap]
            i = candidate[in_gap]  # region before the gap
            dist_to_end = t - rl.end_ns[i]
            dist_to_start = rl.start_ns[i + 1] - t
            snap_to_end = dist_to_end <= dist_to_start
            snap_times = np.where(
                snap_to_end,
                (rl.offset[i + 1] - 1).astype(np.float64) / sfreq,
                rl.offset[i + 1].astype(np.float64) / sfreq,
            )
            result[in_gap] = snap_times

    # --- timestamps slightly before first region ---
    before = times_ns < rl.start_ns[0]
    if before.any():
        close = before & (rl.start_ns[0] - times_ns < sample_period_ns * 10)
        result[close] = rl.offset[0] / sfreq

    # --- timestamps slightly after last region ---
    after = times_ns > rl.end_ns[-1]
    if after.any():
        close = after & (times_ns - rl.end_ns[-1] < sample_period_ns * 10)
        last_sample = (
            rl.offset[-1]
            + (rl.end_ns[-1] - rl.start_ns[-1]) // sample_period_ns
            - 1
        )
        result[close] = last_sample / sfreq

    return result


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def read_raw_dh5(
    fname: str | pathlib.Path,
    cont_ids: list[int] | None = None,
    preload: bool = False,
    verbose: bool | str | int | None = None,
    require_matching_sampling_rate: bool = False,
) -> MneRawDH5:
    """Read a DH5 file as an MNE Raw object.

    Parameters
    ----------
    fname : str | Path
        Path to the .dh5 file.
    cont_ids : list of int | None
        Which CONT groups to include. ``None`` (default) selects all CONT groups.
    preload : bool
        If True, load all data into memory at init. If False, data is read lazily.
    verbose : bool | str | int | None
        MNE verbosity level.
    require_matching_sampling_rate : bool
        If True, raise an error when the selected CONT groups have different sampling
        rates. If False (default), CONT groups whose rate differs from the first are
        silently skipped with a ``DH5SampleRateMismatchWarning``.

    Returns
    -------
    raw : MneRawDH5
    """
    return MneRawDH5(
        fname,
        cont_ids=cont_ids,
        preload=preload,
        verbose=verbose,
        require_matching_sampling_rate=require_matching_sampling_rate,
    )


def read_raw_dh5_per_sfreq(
    fname: str | pathlib.Path,
    preload: bool = False,
    verbose: bool | str | int | None = None,
) -> dict[float, MneRawDH5]:
    """Read a DH5 file, returning one Raw object per unique sampling rate.

    All CONT groups that share a sampling rate are combined into a single Raw
    object (multi-channel).  This is the natural way to read a DH5 file when
    different CONT groups were recorded at different rates.

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
    raws : dict[float, MneRawDH5]
        Mapping from sampling rate (Hz) to Raw object.
    """
    fname = pathlib.Path(fname)
    with DH5File(fname) as dh5:
        groups_by_sfreq = {
            sfreq: [c.id for c in conts]
            for sfreq, conts in dh5.get_cont_groups_by_sfreq().items()
        }
    return {
        sfreq: MneRawDH5(fname, cont_ids=ids, preload=preload, verbose=verbose,
                         require_matching_sampling_rate=True)
        for sfreq, ids in groups_by_sfreq.items()
    }


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
    raws : dict[int, MneRawDH5]
        Mapping from CONT group ID to Raw object.
    """
    fname = pathlib.Path(fname)
    with DH5File(fname) as dh5:
        cont_ids: list[int] = dh5.get_cont_group_ids()
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


def _annotations_from_dh5_continuous(
    fname: str | pathlib.Path,
    start_ns: int,
    sfreq: float,
) -> mne.Annotations:
    """Annotation builder for the normal (continuous, single-region) case.

    Timestamps are converted with the simple formula ``(ts_ns - start_ns) / 1e9``.
    No region lookup or gap handling is needed.
    """
    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    with DH5File(fname) as dh5:
        trialmap = dh5.get_trialmap()
        if trialmap is not None and len(trialmap) > 0:
            for i in range(len(trialmap)):
                t_start = (int(trialmap.start_time_nanoseconds[i]) - start_ns) / 1e9
                t_end = (int(trialmap.end_time_nanoseconds[i]) - start_ns) / 1e9
                dur = t_end - t_start
                if dur > 0:
                    onsets.append(t_start)
                    durations.append(dur)
                    stim_no: int = int(trialmap.trial_type_numbers[i])
                    outcome: int = int(trialmap.trial_outcomes_integer[i])
                    descriptions.append(f"trial/StimNo={stim_no}/Outcome={outcome}")

        events_arr = dh5.get_events_array()
        if events_arr is not None and len(events_arr) > 0:
            for j in range(len(events_arr)):
                t = (int(events_arr["time"][j]) - start_ns) / 1e9
                onsets.append(t)
                durations.append(0.0)
                descriptions.append(f"event/{events_arr['event'][j]}")

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def _annotations_from_dh5_discontinuous(
    fname: str | pathlib.Path,
    region_lookup: _RegionLookup,
    sample_period_ns: int,
    sfreq: float,
) -> mne.Annotations:
    """Annotation builder for the discontinuous (multi-region) case.

    Uses the region lookup to map absolute nanosecond timestamps onto the
    stitched MNE sample timeline and inserts ``BAD_ACQ_SKIP`` annotations at
    each acquisition gap.
    """
    rl = region_lookup

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    with DH5File(fname) as dh5:
        # Region boundaries as BAD annotations
        for i in range(len(rl.start_ns) - 1):
            if rl.start_ns[i + 1] > rl.end_ns[i]:
                boundary_sample: float = float(rl.offset[i + 1]) / sfreq
                onsets.append(boundary_sample)
                durations.append(0.0)
                descriptions.append("BAD_ACQ_SKIP")

        # TRIALMAP
        trialmap = dh5.get_trialmap()
        if trialmap is not None and len(trialmap) > 0:
            start_times = _batch_ns_to_sample_time(
                trialmap.start_time_nanoseconds, rl, sample_period_ns, sfreq
            )
            end_times = _batch_ns_to_sample_time(
                trialmap.end_time_nanoseconds, rl, sample_period_ns, sfreq
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
                events_arr["time"], rl, sample_period_ns, sfreq
            )
            for j in range(len(events_arr)):
                if np.isnan(event_times[j]):
                    continue
                onsets.append(event_times[j])
                durations.append(0.0)
                descriptions.append(f"event/{events_arr['event'][j]}")

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def annotations_from_dh5(
    fname: str | pathlib.Path,
    cont_id: int | None = None,
) -> mne.Annotations:
    """Build MNE Annotations from a DH5 file's TRIALMAP and EV02 events.

    Can be called independently of ``MneRawDH5`` whenever you need the
    annotation objects without constructing a full Raw object.

    For continuous files (all CONT blocks have a single region) the conversion
    is trivial: ``(timestamp_ns - start_ns) / 1e9``.  For discontinuous files
    the full region-lookup machinery is used and ``BAD_ACQ_SKIP`` annotations
    are inserted at acquisition gaps.

    Parameters
    ----------
    fname : str | pathlib.Path
        Path to the DH5 file.
    cont_id : int | None
        ID of the CONT block whose time axis is used as the reference for
        converting nanosecond timestamps to MNE sample times.  When ``None``
        (default) the first CONT block is used, provided all CONT blocks
        share the same start timestamp.  If they do not, a ``DH5Error`` is
        raised and the caller must supply an explicit ``cont_id``.

    Returns
    -------
    annotations : mne.Annotations
        MNE Annotations with trial, event, and (for discontinuous files)
        region-boundary entries.  An empty ``mne.Annotations`` object is
        returned when the file contains no mappable timestamps.

    Raises
    ------
    DH5Error
        If ``cont_id`` is ``None`` and the CONT blocks do not all start at
        the same time.
    """
    fname = pathlib.Path(fname)
    with DH5File(fname) as dh5:
        if cont_id is None:
            if not dh5.cont_blocks_start_simultaneously():
                ids = dh5.get_cont_group_ids()
                raise DH5Error(
                    f"CONT blocks in {fname.name} do not all start at the same time "
                    f"(IDs: {ids}). Pass cont_id=<id> to select a reference block."
                )
            cont_id = dh5.get_cont_group_ids()[0]
        ref_cont = dh5.get_cont_group_by_id(cont_id)
        sample_period_ns = int(ref_cont.sample_period)
        sfreq = 1e9 / sample_period_ns
        start_ns = int(ref_cont.index[0]["time"])

        if not dh5.is_continuous():
            total_samples = ref_cont.n_samples
            region_lookup = _build_region_lookup(ref_cont.index, total_samples, sample_period_ns)
            return _annotations_from_dh5_discontinuous(fname, region_lookup, sample_period_ns, sfreq)

    return _annotations_from_dh5_continuous(fname, start_ns, sfreq)


# ---------------------------------------------------------------------------
# RawDH5 class
# ---------------------------------------------------------------------------
class MneRawDH5(BaseRaw):
    """Raw object for reading DH5 files with lazy loading.

    Parameters
    ----------
    fname : str | Path
        Path to the .dh5 file.
    cont_ids : list of int | None
        Which CONT groups to include. ``None`` (default) selects all CONT groups.
    preload : bool
        If True, load all data into memory.
    verbose : bool | str | int | None
        MNE verbosity level.
    require_matching_sampling_rate : bool
        If True, raise an error when the selected CONT groups have different sampling
        rates. If False (default), mismatched groups are skipped with a warning.
    """

    def __init__(
        self,
        fname: str | pathlib.Path,
        cont_ids: list[int] | None = None,
        preload: bool = False,
        verbose: bool | str | int | None = None,
        require_matching_sampling_rate: bool = False,
    ) -> None:
        fname = pathlib.Path(fname)

        # Collect metadata using dh5io
        dh5 = DH5File(fname)
        if not dh5.get_cont_group_ids():
            raise ValueError(f"No CONT groups found in {fname}")

        selected: list[Cont] = (
            dh5.get_cont_groups_by_ids(cont_ids) if cont_ids is not None
            else dh5.get_cont_groups()
        )

        # Validate / handle differing sampling rates
        target_sfreq: float = 1e9 / selected[0].sample_period
        mismatched = [c for c in selected if 1e9 / c.sample_period != target_sfreq]
        if mismatched:
            names = [f"CONT{c.id} ({1e9 / c.sample_period:.1f} Hz)" for c in mismatched]
            if require_matching_sampling_rate:
                raise ValueError(
                    f"CONT groups have different sampling rates: "
                    f"{', '.join(names)}. Target rate: {target_sfreq:.1f} Hz."
                )
            selected = [c for c in selected if 1e9 / c.sample_period == target_sfreq]
            warnings.warn(
                DH5SampleRateMismatchWarning(
                    f"Skipping CONT groups with non-matching sampling rate: "
                    f"{', '.join(names)}. Target rate: {target_sfreq:.1f} Hz."
                )
            )

        sfreq = target_sfreq

        # Handle differing sample counts
        n_samples_set = set(c.n_samples for c in selected)
        if len(n_samples_set) > 1:
            total_samples: int = min(n_samples_set)
            warnings.warn(
                DH5SampleCountMismatchWarning(
                    f"CONT groups have different sample counts: {n_samples_set}. "
                    f"Using minimum: {total_samples}."
                )
            )
        else:
            total_samples = selected[0].n_samples

        # Build channel map: (cont_id, column_index, calibration_value, ch_name)
        channel_map: list[tuple[int, int, float, str]] = []
        ch_names: list[str] = []
        for c in selected:
            calib = c.calibration
            ch_label: str = c.name if c.name else f"CONT{c.id}"
            for col in range(c.n_channels):
                cal: float = float(calib[col]) if calib is not None else 1.0
                ch_name: str = f"{ch_label}/{col}"
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

        sample_period_ns: int = int(1e9 / sfreq)
        self._sample_period_ns: int = sample_period_ns
        self._sfreq: float = sfreq

        first_cont = dh5.get_cont_group_by_id(selected[0].id)
        self._start_ns: int = int(first_cont.index[0]["time"])

        if dh5.is_continuous():
            self._region_lookup: _RegionLookup | None = None
        else:
            n_regions = first_cont.n_regions
            warnings.warn(
                f"DH5 file contains {n_regions} discontinuous recording regions. "
                "They have been concatenated; gaps are marked as BAD_ACQ_SKIP annotations.",
                DH5DiscontinuousRegionsWarning,
                stacklevel=3,
            )
            self._region_lookup = _build_region_lookup(
                first_cont.index, total_samples, sample_period_ns
            )

        del dh5

        # Add annotations
        if self._region_lookup is None:
            annot = _annotations_from_dh5_continuous(fname, self._start_ns, self._sfreq)
        else:
            annot = _annotations_from_dh5_discontinuous(
                fname, self._region_lookup, self._sample_period_ns, self._sfreq
            )
        if len(annot) > 0:
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
