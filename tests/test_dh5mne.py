# -*- coding: utf-8 -*-
"""Tests for the dh5mne module."""

from __future__ import annotations

import pathlib
import tracemalloc
import warnings
from unittest.mock import MagicMock, patch

import h5py
import mne
import numpy as np
import numpy.typing as npt
import pytest

from dh5io.dh5mne import (
    MneRawDH5,
    _batch_ns_to_sample_time,
    _build_region_lookup,
    _ns_to_sample_time,
    epochs_from_dh5,
    read_raw_dh5,
    read_raw_dh5_per_cont,
)

TEST_FILE: pathlib.Path = pathlib.Path(__file__).parent / "test.dh5"

pytestmark = pytest.mark.skipif(not TEST_FILE.exists(), reason="test.dh5 not present")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_raw_hdf5_data(
    fname: pathlib.Path, cont_id: int, start: int = 0, stop: int = 10
) -> tuple[npt.NDArray[np.int16], float | None]:
    """Read raw int16 data and calibration directly from HDF5."""
    with h5py.File(fname, "r") as f:
        data = f[f"CONT{cont_id}/DATA"][start:stop, :]
        calib = f[f"CONT{cont_id}"].attrs.get("Calibration")
        if calib is not None:
            calib = float(calib[0])
    return data, calib


# ---------------------------------------------------------------------------
# read_raw_dh5 — CONT selection modes
# ---------------------------------------------------------------------------
class TestReadRawDH5:
    def test_default_selects_matching_sfreq(self) -> None:
        raw = read_raw_dh5(TEST_FILE)
        # All 7 CONTs in test.dh5 share 1 kHz
        assert raw.info["nchan"] == 7
        assert raw.info["sfreq"] == 1000.0

    def test_explicit_cont_ids(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1, 1001])
        assert raw.info["nchan"] == 2
        assert raw.ch_names == ["CONT1/0", "CONT1001/0"]

    def test_cont_ids_all(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids="all")
        assert raw.info["nchan"] == 7

    def test_missing_cont_id_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            read_raw_dh5(TEST_FILE, cont_ids=[9999])

    def test_channel_names_include_cont_name(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        assert raw.ch_names == ["CONT1/0"]

    def test_n_times_matches_hdf5(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        with h5py.File(TEST_FILE, "r") as f:
            expected = f["CONT1/DATA"].shape[0]
        assert raw.n_times == expected


# ---------------------------------------------------------------------------
# read_raw_dh5_per_cont
# ---------------------------------------------------------------------------
class TestReadRawPerCont:
    def test_returns_dict_of_raw(self) -> None:
        raws = read_raw_dh5_per_cont(TEST_FILE)
        assert isinstance(raws, dict)
        assert all(isinstance(r, MneRawDH5) for r in raws.values())

    def test_one_raw_per_cont(self) -> None:
        raws = read_raw_dh5_per_cont(TEST_FILE)
        assert set(raws.keys()) == {1, 60, 61, 62, 63, 64, 1001}
        for r in raws.values():
            assert r.info["nchan"] == 1


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class TestCalibration:
    def test_calibrated_channel(self) -> None:
        """CONT1 has calibration — MNE data should equal int16 * cal."""
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        mne_data, _ = raw[:, :10]

        int16_data, cal = _get_raw_hdf5_data(TEST_FILE, cont_id=1, stop=10)
        assert cal is not None
        expected = int16_data[:, 0].astype(np.float64) * cal
        np.testing.assert_allclose(mne_data[0], expected)

    def test_uncalibrated_channel(self) -> None:
        """CONT60 has no calibration — MNE data should equal raw int16 as float."""
        raw = read_raw_dh5(TEST_FILE, cont_ids=[60])
        mne_data, _ = raw[:, :10]

        int16_data, cal = _get_raw_hdf5_data(TEST_FILE, cont_id=60, stop=10)
        assert cal is None
        expected = int16_data[:, 0].astype(np.float64)
        np.testing.assert_allclose(mne_data[0], expected)

    def test_cal_set_in_info(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        with h5py.File(TEST_FILE, "r") as f:
            expected_cal = float(f["CONT1"].attrs["Calibration"][0])
        assert raw.info["chs"][0]["cal"] == pytest.approx(expected_cal)


# ---------------------------------------------------------------------------
# Lazy loading vs preload
# ---------------------------------------------------------------------------
class TestLazyVsPreload:
    def test_lazy_and_preload_match(self) -> None:
        raw_lazy = read_raw_dh5(TEST_FILE, cont_ids=[1], preload=False)
        raw_pre = read_raw_dh5(TEST_FILE, cont_ids=[1], preload=True)
        d1, _ = raw_lazy[:, 500:600]
        d2, _ = raw_pre[:, 500:600]
        np.testing.assert_allclose(d1, d2)

    def test_lazy_not_preloaded(self) -> None:
        raw = read_raw_dh5(TEST_FILE, preload=False)
        assert not raw.preload

    def test_preloaded_flag(self) -> None:
        raw = read_raw_dh5(TEST_FILE, preload=True)
        assert raw.preload

    def test_preload_warns_when_data_exceeds_half_available_memory(self) -> None:
        """ResourceWarning is issued when preload would use >50% of available RAM."""
        mock_mem = MagicMock()
        mock_mem.available = 1  # 1 byte — any real file will exceed half of this
        with patch("dh5io.dh5mne._psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem
            with pytest.warns(ResourceWarning, match="preload=True"):
                read_raw_dh5(TEST_FILE, cont_ids=[1], preload=True)

    def test_preload_no_warning_when_memory_sufficient(self) -> None:
        """No ResourceWarning when available memory is ample."""
        mock_mem = MagicMock()
        mock_mem.available = 2**62  # 4 EiB — always sufficient
        with patch("dh5io.dh5mne._psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem
            with warnings.catch_warnings():
                warnings.simplefilter("error", ResourceWarning)
                read_raw_dh5(TEST_FILE, cont_ids=[1], preload=True)

    def test_preload_no_warning_when_psutil_unavailable(self) -> None:
        """No error or warning when psutil is not installed (graceful fallback)."""
        with patch("dh5io.dh5mne._psutil", None):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ResourceWarning)
                read_raw_dh5(TEST_FILE, cont_ids=[1], preload=True)

    def test_preload_allocates_more_memory_than_lazy(self) -> None:
        """Preloading allocates significantly more memory than lazy loading."""
        tracemalloc.start()

        tracemalloc.clear_traces()
        read_raw_dh5(TEST_FILE, cont_ids=[1], preload=False)
        _, lazy_peak = tracemalloc.get_traced_memory()

        tracemalloc.clear_traces()
        read_raw_dh5(TEST_FILE, cont_ids=[1], preload=True)
        _, preload_peak = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        assert preload_peak > lazy_peak


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
class TestAnnotations:
    @pytest.fixture()
    def raw(self) -> MneRawDH5:
        return read_raw_dh5(TEST_FILE)

    def test_trial_annotations_count(self, raw: MneRawDH5) -> None:
        trials = [
            a for a in raw.annotations if str(a["description"]).startswith("trial/")
        ]
        with h5py.File(TEST_FILE, "r") as f:
            n_in_file = f["TRIALMAP"].shape[0]
        # Trials whose timestamps fall outside all recording regions are skipped;
        # so the annotation count must be <= the TRIALMAP row count, but > 0.
        assert 0 < len(trials) <= n_in_file

    def test_event_annotations_present(self, raw: MneRawDH5) -> None:
        events = [
            a for a in raw.annotations if str(a["description"]).startswith("event/")
        ]
        assert len(events) > 0

    def test_region_boundary_annotations(self, raw: MneRawDH5) -> None:
        boundaries = [
            a for a in raw.annotations if str(a["description"]) == "BAD_region_boundary"
        ]
        with h5py.File(TEST_FILE, "r") as f:
            n_regions = f["CONT1/INDEX"].shape[0]
        # One boundary between each pair of consecutive regions
        assert len(boundaries) == n_regions - 1

    def test_trial_annotation_format(self, raw: MneRawDH5) -> None:
        trial = next(
            a for a in raw.annotations if str(a["description"]).startswith("trial/")
        )
        desc = str(trial["description"])
        assert "StimNo=" in desc
        assert "Outcome=" in desc

    def test_trial_annotation_duration_positive(self, raw: MneRawDH5) -> None:
        for a in raw.annotations:
            if str(a["description"]).startswith("trial/"):
                assert float(a["duration"]) > 0

    def test_events_from_annotations_works(self, raw: MneRawDH5) -> None:
        events, event_id = mne.events_from_annotations(raw, regexp="trial/.*")
        assert events.shape[1] == 3
        assert len(event_id) > 0


# ---------------------------------------------------------------------------
# epochs_from_dh5
# ---------------------------------------------------------------------------
class TestEpochsFromDH5:
    def test_creates_epochs(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1], preload=True)
        # Use short fixed tmax to avoid issues with variable trial lengths
        epochs = epochs_from_dh5(raw, tmin=0.0, tmax=1.0, baseline=None)
        assert isinstance(epochs, mne.Epochs)
        assert len(epochs) > 0

    def test_auto_tmax(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1], preload=True)
        epochs = epochs_from_dh5(raw, baseline=None)
        assert isinstance(epochs, mne.Epochs)


# ---------------------------------------------------------------------------
# Timestamp-to-sample mapping internals
# ---------------------------------------------------------------------------
class TestTimestampMapping:
    @pytest.fixture()
    def single_region(
        self,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]:
        """Single contiguous recording region: 100k samples at 1 kHz."""
        index = np.array([(0, 0)], dtype=[("time", "<i8"), ("offset", "<i8")])
        return _build_region_lookup(
            index, total_samples=100_000, sample_period_ns=1_000_000
        )

    @pytest.fixture()
    def multi_region(
        self,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]:
        """Two regions with a gap in between."""
        index = np.array(
            [(0, 0), (2_000_000_000, 1000)],  # 1s gap after 1000 samples
            dtype=[("time", "<i8"), ("offset", "<i8")],
        )
        return _build_region_lookup(
            index, total_samples=2000, sample_period_ns=1_000_000
        )

    def test_single_region_exact(self, single_region: tuple) -> None:
        rs, re, ro, rn = single_region
        t = _ns_to_sample_time(50_000_000_000, rs, re, ro, 1_000_000, 1000.0)
        assert t == pytest.approx(50.0)

    def test_single_region_start(self, single_region: tuple) -> None:
        rs, re, ro, rn = single_region
        t = _ns_to_sample_time(0, rs, re, ro, 1_000_000, 1000.0)
        assert t == pytest.approx(0.0)

    def test_single_region_end(self, single_region: tuple) -> None:
        rs, re, ro, rn = single_region
        t = _ns_to_sample_time(100_000_000_000, rs, re, ro, 1_000_000, 1000.0)
        assert t == pytest.approx(100.0)

    def test_before_all_regions_with_tolerance(self, single_region: tuple) -> None:
        rs, re, ro, rn = single_region
        t = _ns_to_sample_time(-5_000_000, rs, re, ro, 1_000_000, 1000.0)
        assert t == pytest.approx(0.0)

    def test_far_before_all_regions_returns_none(self, single_region: tuple) -> None:
        rs, re, ro, rn = single_region
        t = _ns_to_sample_time(-100_000_000_000, rs, re, ro, 1_000_000, 1000.0)
        assert t is None

    def test_gap_snaps_to_start_of_next_region(self, multi_region: tuple) -> None:
        rs, re, ro, rn = multi_region
        # Gap: region 0 ends at t=1_000_000_000ns (1000 samples * 1ms),
        #      region 1 starts at t=2_000_000_000ns.
        # Timestamp at 1_999_000_000: 999ms from end of region 0, 1ms from start of region 1
        # → closer to start of region 1 → snaps to offset=1000 → 1000/1000Hz = 1.0s
        gap_time = 1_999_000_000
        t = _ns_to_sample_time(gap_time, rs, re, ro, 1_000_000, 1000.0)
        assert t == pytest.approx(1.0)

    def test_gap_snaps_to_end_of_prev_region(self, multi_region: tuple) -> None:
        rs, re, ro, rn = multi_region
        # Timestamp at 1_001_000_000: 1ms from end of region 0, 999ms from start of region 1
        # → closer to end of region 0 → snaps to last sample of region 0 = offset 999 → 999/1000Hz
        gap_time = 1_001_000_000
        t = _ns_to_sample_time(gap_time, rs, re, ro, 1_000_000, 1000.0)
        assert t == pytest.approx(999 / 1000.0)

    def test_gap_snaps_batch_to_end_of_prev_region(self, multi_region: tuple) -> None:
        rs, re, ro, rn = multi_region
        # Same as above but via _batch_ns_to_sample_time
        times = np.array([1_001_000_000], dtype=np.int64)
        result = _batch_ns_to_sample_time(times, rs, re, ro, 1_000_000, 1000.0)
        assert result[0] == pytest.approx(999 / 1000.0)

    def test_batch_returns_nan_for_unmappable(self, single_region: tuple) -> None:
        rs, re, ro, rn = single_region
        times = np.array([-999_999_999_999, 50_000_000_000], dtype=np.int64)
        result = _batch_ns_to_sample_time(times, rs, re, ro, 1_000_000, 1000.0)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Data access: slicing and multi-CONT
# ---------------------------------------------------------------------------
class TestDataAccess:
    def test_full_channel_slice(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        data, times = raw[:, :100]
        assert data.shape == (1, 100)
        assert times.shape == (100,)

    def test_multi_cont_shape(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1, 1001])
        data, _ = raw[:, :50]
        assert data.shape == (2, 50)

    def test_all_conts_shape(self) -> None:
        raw = read_raw_dh5(TEST_FILE)
        data, _ = raw[:, :20]
        assert data.shape == (7, 20)

    def test_middle_slice(self) -> None:
        """Reading from the middle of the file works."""
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        data, _ = raw[:, 100_000:100_010]
        assert data.shape == (1, 10)
        assert not np.all(data == 0)

    def test_single_sample(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        data, _ = raw[:, 0:1]
        assert data.shape == (1, 1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_string_path(self) -> None:
        raw = read_raw_dh5(str(TEST_FILE), cont_ids=[1])
        assert raw.info["nchan"] == 1

    def test_pathlib_path(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        assert raw.info["nchan"] == 1

    def test_repr_does_not_crash(self) -> None:
        raw = read_raw_dh5(TEST_FILE, cont_ids=[1])
        repr(raw)
        str(raw)
