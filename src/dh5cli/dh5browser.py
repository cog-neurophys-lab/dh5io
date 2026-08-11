"""
DH5 Browser - Interactive data viewer for DH5 files using ephyviewer.

This module provides an interactive browser for DH5 (DAQ-HDF5) files based on ephyviewer.
The browser displays signals, spikes, and events from a single trial.

Usage:
    dh5browser <filename.dh5> [--trial TRIAL_INDEX]

Examples:
    dh5browser mydata.dh5
    dh5browser mydata.dh5 --trial 0
"""

import argparse
import logging
import pathlib
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

try:
    import ephyviewer
except ImportError:
    print("Error: ephyviewer is required for dh5browser")
    print("Install with: pip install ephyviewer")
    sys.exit(1)

import neo
import numpy as np

from dh5io import DH5File
from dh5neo import DH5IO

# Configure logging
logger = logging.getLogger(__name__)

# Time span (in seconds) substituted for sources that cover a single instant.
# ephyviewer's navigation toolbar asserts t_stop > t_start, so a source holding
# a single spike/event would otherwise abort the browser (see DH5MainViewer.add_view).
MIN_TIME_SPAN = 1.0

try:
    import matplotlib

    # Set backend before importing pyplot to avoid premature Qt initialization
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, event colors will not be customized")

try:
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import Signal

    Qt = QtCore.Qt
except ImportError:
    print("Error: PySide6 is required for dh5browser")
    print("Install with: pip install PySide6")
    sys.exit(1)


def has_degenerate_time_range(source) -> bool:
    """True if a data source does not span a positive amount of time.

    Empty sources report t_start == t_stop == 0, and a source holding a single
    spike or event reports t_start == t_stop == <that time>.
    """
    try:
        return not (source.t_stop > source.t_start)
    except (AttributeError, TypeError):
        return False


class DH5MainViewer(ephyviewer.MainViewer):
    """Extended MainViewer that persists window state (dock visibility, positions, sizes)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: We don't restore window state here because viewers haven't been added yet.
        # Call restore_window_state() after adding all viewers.

    def add_view(self, widget, *args, **kwargs):
        """Add a viewer, tolerating sources that cover a single instant.

        MainViewer.add_view() forwards the source time range to the navigation
        toolbar, which asserts t_stop > t_start. Files without continuous data
        can produce sources spanning zero time, which would abort the browser.
        """
        source = getattr(widget, "source", None)
        name = getattr(widget, "name", type(widget).__name__)

        if source is not None and has_degenerate_time_range(source):
            padded_stop = source.t_start + MIN_TIME_SPAN
            logger.debug(
                f"Padding zero-length time range of '{name}': "
                f"t_stop {source.t_stop} -> {padded_stop}"
            )
            # In-memory sources store the value; computed t_stop properties ignore this
            source._t_stop = padded_stop

        try:
            super().add_view(widget, *args, **kwargs)
        except AssertionError:
            # The dock widget is registered before the navigation range is updated,
            # so the viewer itself is usable - only the time range update failed.
            logger.warning(
                f"Viewer '{name}' has no usable time range; "
                "keeping the current navigation range"
            )

    def restore_window_state(self):
        """Restore window geometry and dock widget state from settings.

        This should be called AFTER all viewers have been added to the window.
        """
        if self.settings_name is not None:
            # Restore main window geometry
            geometry = self.settings.value("window_geometry")
            if geometry is not None:
                try:
                    self.restoreGeometry(geometry)
                    logger.debug("Restored window geometry from settings")
                except Exception as e:
                    logger.debug(f"Could not restore window geometry: {e}")

            # Restore dock widget state (visibility, positions, sizes)
            # This must be done after all dock widgets (viewers) have been added
            state = self.settings.value("window_state")
            if state is not None:
                try:
                    self.restoreState(state)
                    logger.debug("Restored window state from settings")
                except Exception as e:
                    logger.debug(f"Could not restore window state: {e}")

    def save_all_settings(self):
        """Save all viewer settings plus window geometry and state."""
        # Call parent implementation to save viewer parameters
        super().save_all_settings()

        # Additionally save window geometry and dock state
        if self.settings_name is not None:
            self.settings.setValue("window_geometry", self.saveGeometry())
            self.settings.setValue("window_state", self.saveState())
            logger.debug("Saved window geometry and state to settings")


class SegmentCache:
    """Cache for loaded Neo segments to avoid re-reading from disk with prefetching support."""

    def __init__(
        self,
        reader: DH5IO,
        max_cache_size: int = 10,
        prefetch_count: int = 3,
        nb_segments: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        reader : DH5IO
            The DH5 reader instance
        max_cache_size : int
            Maximum number of segments to keep in cache
        prefetch_count : int
            Number of trials to prefetch ahead (default: 3)
        nb_segments : int, optional
            Total number of segments (for bounds checking during prefetch)
        """
        self.reader = reader
        self.max_cache_size = max_cache_size
        self.prefetch_count = prefetch_count
        self.nb_segments = nb_segments
        self._cache: Dict[int, Tuple[neo.Segment, float]] = {}
        self._access_order: List[int] = []
        self._cache_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="prefetch"
        )
        self._prefetch_futures: Dict[int, Future] = {}  # Track ongoing prefetches

    def _load_segment_internal(self, segment_index: int) -> neo.Segment:
        """Internal method to load a segment from disk (thread-safe)."""
        step_start = time.perf_counter()
        segment = self.reader.read_segment(
            block_index=0, seg_index=segment_index, lazy=False
        )
        logger.debug(
            f"  read_segment({segment_index}): {time.perf_counter() - step_start:.3f}s"
        )
        return segment

    def _prefetch_segment(self, segment_index: int) -> None:
        """Prefetch a segment in the background (called by executor thread)."""
        with self._cache_lock:
            # Check if already cached
            if segment_index in self._cache:
                logger.debug(
                    f"Segment {segment_index} already in cache, skipping prefetch"
                )
                return

            # Check if cache is full
            if len(self._cache) >= self.max_cache_size:
                logger.debug(
                    f"Cache full, skipping prefetch of segment {segment_index}"
                )
                return

        # Load segment (outside lock to avoid blocking main thread)
        logger.debug(f"Prefetching segment {segment_index} in background")
        segment = self._load_segment_internal(segment_index)

        # Add to cache
        with self._cache_lock:
            # Double-check it wasn't added while we were loading
            if segment_index not in self._cache:
                # Only add if cache still has room
                if len(self._cache) < self.max_cache_size:
                    self._cache[segment_index] = (segment, time.time())
                    self._access_order.append(segment_index)
                    logger.debug(f"Prefetched segment {segment_index} added to cache")
                else:
                    logger.debug(
                        f"Cache filled during prefetch, discarding segment {segment_index}"
                    )

    def _start_prefetch(self, start_index: int) -> None:
        """Start prefetching trials ahead of the given index."""
        if self.prefetch_count <= 0 or self.nb_segments is None:
            return

        for i in range(1, self.prefetch_count + 1):
            next_index = start_index + i
            if next_index >= self.nb_segments:
                break

            # Skip if already cached or being prefetched
            with self._cache_lock:
                if next_index in self._cache or next_index in self._prefetch_futures:
                    continue

            # Submit prefetch task
            logger.debug(f"Scheduling prefetch for segment {next_index}")
            future = self._executor.submit(self._prefetch_segment, next_index)
            self._prefetch_futures[next_index] = future

            # Clean up completed futures
            completed = [
                idx for idx, fut in self._prefetch_futures.items() if fut.done()
            ]
            for idx in completed:
                del self._prefetch_futures[idx]

    def get_segment(self, segment_index: int) -> neo.Segment:
        """
        Get a segment, loading from disk if not cached.

        Parameters
        ----------
        segment_index : int
            Index of segment to retrieve

        Returns
        -------
        neo.Segment
            The requested segment
        """
        with self._cache_lock:
            if segment_index in self._cache:
                logger.debug(f"Cache hit for segment {segment_index}")
                # Update access order
                self._access_order.remove(segment_index)
                self._access_order.append(segment_index)
                segment = self._cache[segment_index][0]
            else:
                # Cache miss - load from disk
                logger.debug(
                    f"Cache miss for segment {segment_index}, loading from disk"
                )
                segment = self._load_segment_internal(segment_index)

                # Add to cache
                self._cache[segment_index] = (segment, time.time())
                self._access_order.append(segment_index)

                # Evict oldest if cache is full
                if len(self._cache) > self.max_cache_size:
                    oldest = self._access_order.pop(0)
                    del self._cache[oldest]
                    logger.debug(f"Evicted segment {oldest} from cache (LRU)")

        # Start prefetching next trials (outside lock)
        self._start_prefetch(segment_index)

        return segment

    def clear(self) -> None:
        """Clear the cache and shutdown executor."""
        with self._cache_lock:
            self._cache.clear()
            self._access_order.clear()
            logger.debug("Cache cleared")

    def shutdown(self) -> None:
        """Shutdown the prefetch executor."""
        logger.debug("Shutting down prefetch executor")
        self._executor.shutdown(wait=False)


class TrialNavigationWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Widget for navigating between trials with Previous/Next buttons and direct trial selection."""

    trial_changed = Signal(int)  # Signal emitted when trial changes
    time_changed = Signal(float, float)  # Required by ephyviewer (t_start, t_stop)

    def __init__(
        self,
        nb_trials: int,
        initial_trial: int = 0,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """
        Parameters
        ----------
        nb_trials : int
            Total number of trials
        initial_trial : int
            Initial trial index
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.name = "Trial Navigation"  # Required by ephyviewer
        self.source = (
            None  # Required by ephyviewer (no data source for navigation widget)
        )
        self.nb_trials = nb_trials
        self.current_trial = initial_trial

        # Create layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Previous button
        self.prev_btn = QtWidgets.QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self._on_previous)
        layout.addWidget(self.prev_btn)

        # Trial selector label
        layout.addWidget(QtWidgets.QLabel("Trial:"))

        # Trial spinbox (editable, allows jumping to any trial)
        self.trial_spinbox = QtWidgets.QSpinBox()
        self.trial_spinbox.setMinimum(1)
        self.trial_spinbox.setMaximum(nb_trials)
        self.trial_spinbox.setValue(initial_trial + 1)  # Display as 1-based
        self.trial_spinbox.setMinimumWidth(80)
        self.trial_spinbox.valueChanged.connect(self._on_trial_changed)
        layout.addWidget(self.trial_spinbox)

        # Total trials label
        self.total_label = QtWidgets.QLabel(f"of {self.nb_trials}")
        layout.addWidget(self.total_label)

        # Next button
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._on_next)
        layout.addWidget(self.next_btn)

        # Stretch to keep buttons compact
        layout.addStretch()

        # Update button states
        self._update_buttons()

    def _update_spinbox(self) -> None:
        """Update the trial spinbox value without triggering signal."""
        # Temporarily block signals to avoid triggering _on_trial_changed
        self.trial_spinbox.blockSignals(True)
        self.trial_spinbox.setValue(self.current_trial + 1)
        self.trial_spinbox.blockSignals(False)

    def _on_trial_changed(self, value: int) -> None:
        """Handle trial spinbox value change (user typed or used arrows)."""
        new_trial = value - 1  # Convert from 1-based to 0-based
        if 0 <= new_trial < self.nb_trials and new_trial != self.current_trial:
            self.current_trial = new_trial
            self._update_buttons()
            self.trial_changed.emit(self.current_trial)

    def _update_buttons(self) -> None:
        """Enable/disable buttons based on current trial."""
        self.prev_btn.setEnabled(self.current_trial > 0)
        self.next_btn.setEnabled(self.current_trial < self.nb_trials - 1)

    def _on_previous(self) -> None:
        """Handle Previous button click."""
        if self.current_trial > 0:
            self.current_trial -= 1
            self._update_spinbox()
            self._update_buttons()
            self.trial_changed.emit(self.current_trial)

    def _on_next(self) -> None:
        """Handle Next button click."""
        if self.current_trial < self.nb_trials - 1:
            self.current_trial += 1
            self._update_spinbox()
            self._update_buttons()
            self.trial_changed.emit(self.current_trial)

    def set_trial(self, trial_index: int) -> None:
        """Programmatically set the current trial."""
        if 0 <= trial_index < self.nb_trials:
            self.current_trial = trial_index
            self._update_spinbox()
            self._update_buttons()

    def get_settings(self) -> Dict[str, int]:
        """Get widget settings for ephyviewer persistence.

        Note: We intentionally don't persist the trial selection.
        Users expect to start at trial 0 (or the --trial argument) each time,
        not to resume where they left off in a previous session.
        """
        return {}

    def set_settings(self, settings: Dict[str, int]) -> None:
        """Set widget settings for ephyviewer persistence.

        Note: We don't restore trial selection from saved settings.
        """
        pass  # Intentionally don't restore trial - use initial_trial instead

    def seek(self, t: float) -> None:
        """Dummy seek method required by ephyviewer (navigation widget doesn't need to seek)."""
        pass  # Navigation widget doesn't display time-based data


class TrialInfoWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Widget to display trial metadata (TrialNo, StimNo, Outcome)."""

    time_changed = Signal(float, float)  # Required by ephyviewer (t_start, t_stop)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.name = "Trial Info"  # Required by ephyviewer
        self.source = None  # Required by ephyviewer (no data source)

        # Create layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Info label
        self.label = QtWidgets.QLabel("Trial Info: No trial loaded")
        self.label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.label)

        # Stretch to keep label compact
        layout.addStretch()

    def update_trial_info(self, trial_no: int, stim_no: int, outcome: int) -> None:
        """Update the displayed trial information.

        Parameters
        ----------
        trial_no : int
            Trial number from TRIALMAP
        stim_no : int
            Stimulus/trial type number from TRIALMAP
        outcome : int
            Outcome code from TRIALMAP
        """
        self.label.setText(
            f"Trial Info: TrialNo={trial_no}, TrialTypeNo={stim_no}, Outcome={outcome}"
        )

    def get_settings(self) -> Dict[str, int]:
        """Get widget settings for ephyviewer persistence."""
        return {}

    def set_settings(self, settings: Dict[str, int]) -> None:
        """Set widget settings for ephyviewer persistence."""
        pass

    def seek(self, t: float) -> None:
        """Dummy seek method required by ephyviewer (info widget doesn't need to seek)."""
        pass


class CombinedAnalogSignalSource:
    """
    Custom ephyviewer source that combines multiple analog signals into one multi-channel source.
    Preserves the original channel names from Neo array_annotations.
    Supports updating signals dynamically for efficient trial switching.
    """

    def __init__(self, analog_signals: List[neo.AnalogSignal]) -> None:
        """
        Parameters
        ----------
        analog_signals : list of neo.AnalogSignal
            List of analog signals to combine
        """
        init_start: float = time.perf_counter()

        self.signals: List[neo.AnalogSignal] = analog_signals
        self.type: str = "analogsignal"
        self.with_scatter: bool = False

        self._process_signals(analog_signals)

        logger.debug(
            f"CombinedAnalogSignalSource.__init__ total: {time.perf_counter() - init_start:.3f}s"
        )

    def _process_signals(self, analog_signals: List[neo.AnalogSignal]) -> None:
        """Process analog signals and update internal data structures."""
        # All signals should have the same sampling rate and t_start
        param_start = time.perf_counter()
        self.sample_rate: float = float(analog_signals[0].sampling_rate.magnitude)  # type: ignore[union-attr]
        self.t_start: float = float(analog_signals[0].t_start.magnitude)
        self.t_stop: float = float(analog_signals[0].t_stop.magnitude)
        logger.debug(f"  Extract parameters: {time.perf_counter() - param_start:.3f}s")

        # Collect channel names from array_annotations
        names_start = time.perf_counter()
        self.channel_names: List[str] = []
        for sig in analog_signals:
            if (
                hasattr(sig, "array_annotations")
                and "channel_names" in sig.array_annotations
            ):
                # Extract channel names from array_annotations
                for ch_name in sig.array_annotations["channel_names"]:
                    self.channel_names.append(str(ch_name))
            else:
                # Fallback to signal name
                for i in range(sig.shape[1]):
                    self.channel_names.append(f"{sig.name}/{i}")

        self.nb_channel = len(self.channel_names)
        logger.debug(
            f"  Collect channel names ({self.nb_channel} channels): {time.perf_counter() - names_start:.3f}s"
        )

        # Concatenate all signals along channel axis
        concat_start = time.perf_counter()
        signal_arrays = [sig.magnitude for sig in analog_signals]
        logger.debug(
            f"    Extract magnitude arrays: {time.perf_counter() - concat_start:.3f}s"
        )

        concat2_start = time.perf_counter()
        self._data: np.ndarray = np.concatenate(signal_arrays, axis=1)
        self.nb_channel: int = len(self.channel_names)
        logger.debug(f"    np.concatenate: {time.perf_counter() - concat2_start:.3f}s")
        logger.debug(
            f"  Total concatenation: {time.perf_counter() - concat_start:.3f}s"
        )

    def update_signals(self, analog_signals: List[neo.AnalogSignal]) -> None:
        """
        Update the source with new analog signals.
        This allows reusing the same source object with different data.

        Parameters
        ----------
        analog_signals : list of neo.AnalogSignal
            New list of analog signals to display
        """
        self.signals = analog_signals
        self._process_signals(analog_signals)

    def get_shape(self) -> Tuple[int, int]:
        shape = self._data.shape
        return (shape[0], shape[1])

    def get_length(self) -> int:
        return self._data.shape[0]

    def get_chunk(
        self, i_start: Optional[int] = None, i_stop: Optional[int] = None
    ) -> np.ndarray:
        """Get a chunk of data."""
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._data.shape[0]
        return self._data[i_start:i_stop, :]

    def get_channel_name(
        self, chan: Optional[int] = None, channel_index: Optional[int] = None
    ) -> str:
        """Get the name of a specific channel."""
        # Accept both 'chan' (used by ephyviewer) and 'channel_index'
        idx = chan if chan is not None else channel_index
        if idx is None:
            raise ValueError("Either chan or channel_index must be provided")
        return self.channel_names[idx]

    def index_to_time(self, index: int) -> float:
        """Convert sample index to time."""
        return self.t_start + index / self.sample_rate

    def time_to_index(self, time: float) -> int:
        """Convert time to sample index."""
        return int((time - self.t_start) * self.sample_rate)


def create_global_event_epoch_sources(dh5_file: pathlib.Path):
    """
    Create a single ephyviewer epoch source from ALL events and trials in the DH5 file.

    This bypasses the Neo segment filtering to load all events and trials globally,
    so they can be displayed across all trials without reloading.
    All event types and trial epochs are combined into a single source with multiple channels.

    Parameters
    ----------
    dh5_file : pathlib.Path
        Path to the DH5 file

    Returns
    -------
    tuple or None
        (source, name) tuple with the combined epoch source, or None if no events
    """
    # Open DH5 file directly to access raw events and trials
    dh5 = DH5File(dh5_file, "r")

    # Create one channel per event type, all in a single epoch source
    all_epochs = []

    # First, add ALL trial epochs as the first channel
    trialmap = dh5.get_trialmap()
    if trialmap is not None and len(trialmap) > 0:
        trial_times = []
        trial_durations = []
        trial_labels = []

        for trial in trialmap:
            start_time = trial["StartTime"] / 1e9  # Convert ns to seconds
            end_time = trial["EndTime"] / 1e9  # Convert ns to seconds
            duration = end_time - start_time

            trial_times.append(start_time)
            trial_durations.append(duration)
            trial_labels.append(f"Trial_{trial['TrialNo']}_Stim_{trial['StimNo']}")

        all_epochs.append(
            {
                "name": "trials",
                "time": np.array(trial_times, dtype=np.float64),
                "duration": np.array(trial_durations, dtype=np.float64),
                "label": np.array(trial_labels, dtype="U"),
            }
        )
        logger.debug(f"Added trial epochs channel with {len(trial_times)} trials")

    # Get all events from the file
    all_events = dh5.get_events_array()

    if all_events is None or len(all_events) == 0:
        logger.debug("No events found in DH5 file")
        if not all_epochs:
            return None
    else:
        logger.debug(f"Found {len(all_events)} total events in file")

        # Get unique event codes (positive values only)
        event_codes = all_events["event"]
        unique_codes = sorted(set(abs(code) for code in event_codes if code != 0))

        logger.debug(f"Unique event codes: {unique_codes}")

        for event_code in unique_codes:
            # Find onsets (positive) and offsets (negative) for this event type
            onsets = []
            offsets = []

            for t, code in zip(all_events["time"], all_events["event"]):
                if abs(code) == event_code:
                    if code > 0:
                        onsets.append(t)
                    elif code < 0:
                        offsets.append(t)

            onsets = sorted(onsets)
            offsets = sorted(offsets)

            if not onsets:
                continue

            # Pair onsets with offsets
            epoch_times = []
            epoch_durations = []
            epoch_labels = []
            onset_idx = 0
            offset_idx = 0

            while onset_idx < len(onsets):
                onset_time = onsets[onset_idx]

                # Find next offset after this onset
                while offset_idx < len(offsets) and offsets[offset_idx] <= onset_time:
                    offset_idx += 1

                if offset_idx < len(offsets):
                    # Found matching offset
                    offset_time = offsets[offset_idx]
                    duration = offset_time - onset_time

                    if duration > 0:
                        epoch_times.append(onset_time / 1e9)  # Convert ns to seconds
                        epoch_durations.append(duration / 1e9)  # Convert ns to seconds
                        epoch_labels.append(f"Event_{event_code}")
                    offset_idx += 1
                else:
                    # No matching offset - use default 100ms duration
                    epoch_times.append(onset_time / 1e9)  # Convert ns to seconds
                    epoch_durations.append(0.1)  # 100ms default
                    epoch_labels.append(f"Event_{event_code}")

                onset_idx += 1

            if epoch_times:
                # Add this event type as a channel
                all_epochs.append(
                    {
                        "name": f"Event_{event_code}",
                        "time": np.array(epoch_times, dtype=np.float64),
                        "duration": np.array(epoch_durations, dtype=np.float64),
                        "label": np.array(epoch_labels, dtype="U"),
                    }
                )
                logger.debug(
                    f"Added channel for Event_{event_code} with {len(epoch_times)} epochs"
                )

    if not all_epochs:
        return None

    # Create single ephyviewer InMemoryEpochSource with all event types and trials as channels
    source = ephyviewer.InMemoryEpochSource(all_epochs=all_epochs)
    logger.info(
        f"Created combined epoch source with {len(all_epochs)} channels (trials + events)"
    )

    return (source, "Trials and Events")


def create_browser(
    dh5_file: pathlib.Path,
    trial_index: int = 0,
    cache_size: int = 10,
    prefetch_count: int = 3,
) -> Tuple[DH5MainViewer, str, SegmentCache]:
    """
    Create an ephyviewer MainViewer window for a DH5 file with trial navigation.

    Parameters
    ----------
    dh5_file : pathlib.Path
        Path to the DH5 file to open
    trial_index : int, optional
        Index of the trial to display (default: 0)
    cache_size : int, optional
        Maximum number of trials to cache (default: 10)
    prefetch_count : int, optional
        Number of trials to prefetch ahead in background (default: 3)

    Returns
    -------
    DH5MainViewer
        The configured main viewer window with state persistence
    str
        The filename
    SegmentCache
        The cache instance (for cleanup)
    """
    overall_start = time.perf_counter()

    # Load DH5 file using Neo
    logger.info(f"Loading DH5 file: {dh5_file}")
    step_start = time.perf_counter()
    reader = DH5IO(dh5_file)
    logger.debug(f"  DH5IO initialization: {time.perf_counter() - step_start:.3f}s")

    # Get segment count without reading all data
    reader.parse_header()
    nb_segments = reader.segment_count(0)

    if nb_segments == 0:
        print("Error: No trials found in DH5 file")
        sys.exit(1)

    if trial_index >= nb_segments:
        print(f"Error: Trial index {trial_index} out of range")
        print(f"File contains {nb_segments} trial(s)")
        sys.exit(1)

    # Create segment cache with prefetching
    cache = SegmentCache(
        reader,
        max_cache_size=cache_size,
        prefetch_count=prefetch_count,
        nb_segments=nb_segments,
    )

    # Load initial segment
    segment = cache.get_segment(trial_index)

    logger.info(f"Loading trial {trial_index} of {nb_segments}")
    logger.debug(f"  Analog signals: {len(segment.analogsignals)}")
    logger.debug(f"  Spike trains: {len(segment.spiketrains)}")
    logger.debug(f"  Events: {len(segment.events)}")
    logger.debug(f"  Epochs: {len(segment.epochs)}")

    # Report data sizes
    if segment.analogsignals:
        total_samples = sum(sig.shape[0] for sig in segment.analogsignals)
        total_channels = sum(sig.shape[1] for sig in segment.analogsignals)
        total_mb = sum(sig.magnitude.nbytes for sig in segment.analogsignals) / (
            1024**2
        )
        logger.info(
            f"  Total analog data: {total_samples:,} samples × {total_channels} channels = {total_mb:.1f} MB"
        )

    # Create the main viewer window with state persistence
    logger.debug("Creating MainViewer window")
    step_start = time.perf_counter()
    win = DH5MainViewer(
        debug=False,
        show_auto_scale=True,
        show_global_xsize=True,
        settings_name=f"dh5browser_{dh5_file.stem}",
    )
    # Set window title to show filename
    win.setWindowTitle(f"DH5 Browser - {dh5_file.name}")
    logger.debug(f"  MainViewer created: {time.perf_counter() - step_start:.3f}s")

    # Create global event epoch source (loaded once, not per trial)
    logger.debug("Loading global event epoch source")
    step_start = time.perf_counter()
    global_event_epoch_source = create_global_event_epoch_sources(dh5_file)
    logger.debug(
        f"  Global event epochs loaded: {time.perf_counter() - step_start:.3f}s"
    )
    if global_event_epoch_source:
        logger.debug("  Found event epoch source with multiple channels")

    # Store references to reusable viewer widgets
    trace_viewer_widget = None
    spike_viewer_widgets = []
    epoch_viewer_widgets = []

    trial_info_widget = None

    # Track whether this is the initial load (for auto-scale behavior)
    is_initial_load = True

    # Function to load and display a trial
    def load_trial(trial_idx: int) -> None:
        """Load and display a specific trial by updating data sources in existing widgets."""
        nonlocal \
            trace_viewer_widget, \
            spike_viewer_widgets, \
            epoch_viewer_widgets, \
            trial_info_widget, \
            is_initial_load

        load_start = time.perf_counter()
        logger.info(f"Loading trial {trial_idx}...")

        # Get segment from cache
        seg = cache.get_segment(trial_idx)

        # Get sources from the Neo segment
        logger.debug("Creating ephyviewer sources from Neo segment")
        step_start = time.perf_counter()
        sources = ephyviewer.get_sources_from_neo_segment(seg)
        logger.debug(
            f"  get_sources_from_neo_segment(): {time.perf_counter() - step_start:.3f}s"
        )

        logger.debug(f"  Created {len(sources)} data source groups:")
        for name, source_list in sources.items():
            if isinstance(source_list, list):
                logger.debug(f"    - {name} ({len(source_list)} sources)")
            else:
                logger.debug(f"    - {name} ({type(source_list).__name__})")

        # Add viewers for different data types
        view_count = 0

        # Extract individual sources from the lists. get_sources_from_neo_segment()
        # always returns a spike source, even when the segment has no spike trains,
        # so drop the ones without channels rather than building empty viewers.
        spike_sources = [
            source for source in sources.get("spike", []) if source.nb_channel > 0
        ]
        # No longer need to extract trial epochs - they're loaded globally

        # Combine all analog signals into a single multi-channel source
        initial_t_start = None
        initial_t_stop = None
        if seg.analogsignals:
            logger.info(
                f"Combining {len(seg.analogsignals)} analog signals into single viewer"
            )

            # Reuse or create TraceViewer
            if trace_viewer_widget is None:
                # Create new viewer on first load
                step_start = time.perf_counter()
                combined_source = CombinedAnalogSignalSource(seg.analogsignals)
                logger.debug(
                    f"  CombinedAnalogSignalSource created: {time.perf_counter() - step_start:.3f}s"
                )
                logger.debug(f"  Shape: {combined_source.get_shape()}")
                logger.debug(f"  Channels: {combined_source.nb_channel}")
                logger.debug(
                    f"  Memory: {combined_source._data.nbytes / (1024**2):.1f} MB"
                )

                step_start = time.perf_counter()
                trace_view = ephyviewer.TraceViewer(
                    source=combined_source, name="All Signals"
                )
                logger.debug(
                    f"  TraceViewer created: {time.perf_counter() - step_start:.3f}s"
                )

                trace_view.params["scale_mode"] = "same_for_all"
                trace_view.params["display_labels"] = True
                trace_view.params["xsize"] = 10.0  # Show 10 seconds initially

                step_start = time.perf_counter()
                win.add_view(trace_view)
                logger.debug(
                    f"  TraceViewer added to window: {time.perf_counter() - step_start:.3f}s"
                )
                trace_viewer_widget = trace_view
                view_count += 1
            else:
                # Update existing viewer's source with new data
                logger.debug("Reusing existing TraceViewer, updating source data")
                trace_viewer_widget.source.update_signals(seg.analogsignals)
                trace_viewer_widget.refresh()

            # Store the time range from the actual source being used by the viewer
            # trace_viewer_widget is guaranteed non-None here
            assert trace_viewer_widget is not None
            viewer_source = trace_viewer_widget.source
            initial_t_start = viewer_source.t_start  # type: ignore[attr-defined]
            initial_t_stop = viewer_source.t_stop  # type: ignore[attr-defined]
        else:
            logger.info(
                f"Trial {trial_idx} has no continuous data; "
                "only events/epochs will be shown"
            )

        # Reuse or create spike train viewers
        logger.debug("Updating spike/event/epoch viewers")
        step_start = time.perf_counter()
        if spike_sources:
            for i, source in enumerate(spike_sources):
                if i < len(spike_viewer_widgets):
                    # Reuse existing viewer - update source data in-place
                    spike_viewer_widgets[i].source.all = source.all
                    spike_viewer_widgets[i].refresh()
                else:
                    # Create new viewer
                    spike_view = ephyviewer.SpikeTrainViewer(
                        source=source, name=f"Spike Trains {i + 1}"
                    )
                    spike_view.params["xsize"] = 10.0  # Match trace viewer
                    win.add_view(spike_view)
                    spike_viewer_widgets.append(spike_view)
                    view_count += 1

        # NOTE: Epoch viewers are NOT updated here since we use global epoch sources
        # that include both trials and events and don't change when switching trials

        if view_count > 0:
            logger.debug(
                f"  Added {view_count} new viewers: {time.perf_counter() - step_start:.3f}s"
            )

        if (
            trace_viewer_widget is None
            and len(spike_viewer_widgets) == 0
            and len(epoch_viewer_widgets) == 0
        ):
            logger.warning("No viewable data found in segment")

        # Auto-scale the trace viewers, but only on initial load if no saved settings exist
        should_auto_scale = False
        if is_initial_load:
            # Check if saved settings exist for the trace viewer
            has_saved_settings = False
            if win.settings_name is not None and trace_viewer_widget is not None:
                viewer_key = f"viewer_{trace_viewer_widget.name}"
                saved_params = win.settings.value(viewer_key)
                has_saved_settings = saved_params is not None

                logger.debug(f"Checking for saved settings: key='{viewer_key}'")
                logger.debug(f"  Saved settings exist: {has_saved_settings}")

                if has_saved_settings and trace_viewer_widget is not None:
                    # Log current ylim values after ephyviewer restored them
                    try:
                        ylim_min = trace_viewer_widget.params["ylim_min"]
                        ylim_max = trace_viewer_widget.params["ylim_max"]
                        logger.debug(
                            f"  Current ylim after restore: min={ylim_min}, max={ylim_max}"
                        )
                    except (KeyError, TypeError):
                        logger.debug("  Could not read ylim values from params")

            # Only auto-scale if no saved settings exist
            should_auto_scale = not has_saved_settings
            if not should_auto_scale:
                logger.info(
                    "Skipping auto-scale: using saved y-axis limits from settings"
                )
            else:
                logger.debug("No saved settings found, will auto-scale")

        if should_auto_scale:
            logger.debug("Auto-scaling viewers")
            step_start = time.perf_counter()
            win.auto_scale()
            logger.debug(f"  auto_scale(): {time.perf_counter() - step_start:.3f}s")

            # Log ylim values after auto-scale
            if trace_viewer_widget is not None:
                try:
                    ylim_min = trace_viewer_widget.params["ylim_min"]
                    ylim_max = trace_viewer_widget.params["ylim_max"]
                    logger.debug(
                        f"  Y-axis limits after auto-scale: min={ylim_min}, max={ylim_max}"
                    )
                except (KeyError, TypeError):
                    logger.debug("  Could not read ylim values after auto-scale")

        # Update navigation toolbar's global time range and seek to the new trial's data start time
        if initial_t_start is not None:
            # Update the navigation toolbar's global time limits
            # This is crucial - otherwise the toolbar keeps the old trial's time range
            # set_start_stop() asserts t_stop > t_start, so pad empty signals
            if initial_t_stop is None or initial_t_stop <= initial_t_start:
                initial_t_stop = initial_t_start + MIN_TIME_SPAN
            win.navigation_toolbar.set_start_stop(
                initial_t_start, initial_t_stop, seek=False
            )

            logger.info(f"Seeking all viewers to t={initial_t_start}")
            step_start = time.perf_counter()
            # Explicitly seek each viewer to the new trial's start time
            if trace_viewer_widget is not None:
                trace_viewer_widget.seek(initial_t_start)
            for spike_viewer in spike_viewer_widgets:
                spike_viewer.seek(initial_t_start)
            for epoch_viewer in epoch_viewer_widgets:
                epoch_viewer.seek(initial_t_start)
            # Also call MainViewer's seek to synchronize
            win.seek(initial_t_start)
            logger.info(
                f"  seek({initial_t_start}): {time.perf_counter() - step_start:.3f}s"
            )

        load_elapsed = time.perf_counter() - load_start
        logger.info(f"Trial {trial_idx} loaded in {load_elapsed:.3f}s")

        # Update trial info widget with metadata from segment annotations
        if trial_info_widget is not None and hasattr(seg, "annotations"):
            trial_no = seg.annotations.get("trial_no", "N/A")
            stim_no = seg.annotations.get("stim_no", "N/A")
            outcome = seg.annotations.get("outcome", "N/A")
            trial_info_widget.update_trial_info(trial_no, stim_no, outcome)

        # Mark that initial load is complete
        is_initial_load = False

    # Load initial trial
    load_trial(trial_index)

    # Add global epoch viewer with ALL trials and events (created once, never reloaded)
    if global_event_epoch_source and global_event_epoch_source[0].nb_channel > 0:
        logger.debug("Adding global epoch viewer with trials and events")
        source, source_name = global_event_epoch_source

        epoch_view = ephyviewer.EpochViewer(source=source, name=source_name)
        epoch_view.params["xsize"] = 10.0  # Match trace viewer
        win.add_view(epoch_view)

        # Apply colors to distinguish event types
        if HAS_MATPLOTLIB:
            import matplotlib.cm
            import matplotlib.colors

            n_channels = source.nb_channel
            cmap = matplotlib.cm.get_cmap("tab10", n_channels)

            epoch_view.by_channel_params.blockSignals(True)
            for c in range(n_channels):
                color = [
                    int(e * 255)
                    for e in matplotlib.colors.ColorConverter().to_rgb(cmap(c))
                ]
                epoch_view.by_channel_params[f"ch{c}", "color"] = color
            epoch_view.by_channel_params.blockSignals(False)
            epoch_view.refresh()

        # Add to list for synchronization
        epoch_viewer_widgets.append(epoch_view)

        # Sync with other viewers if trace viewer exists
        if trace_viewer_widget is not None and hasattr(
            trace_viewer_widget.source, "t_start"
        ):
            epoch_view.seek(trace_viewer_widget.source.t_start)

    # Add trial info widget
    logger.debug("Adding trial info widget")
    trial_info_widget = TrialInfoWidget()
    win.add_view(trial_info_widget, location="top", orientation="horizontal")
    # Update with initial trial info
    initial_segment = cache.get_segment(trial_index)
    if hasattr(initial_segment, "annotations"):
        trial_no = initial_segment.annotations.get("trial_no", "N/A")
        stim_no = initial_segment.annotations.get("stim_no", "N/A")
        outcome = initial_segment.annotations.get("outcome", "N/A")
        trial_info_widget.update_trial_info(trial_no, stim_no, outcome)

    # Add navigation widget if there are multiple trials
    if nb_segments > 1:
        logger.debug(f"Adding navigation widget ({nb_segments} trials)")
        nav_widget = TrialNavigationWidget(nb_segments, initial_trial=trial_index)

        # Connect trial change signal
        def on_trial_changed(new_trial_index: int) -> None:
            logger.info(f"Switching to trial {new_trial_index}")
            load_trial(new_trial_index)

        nav_widget.trial_changed.connect(on_trial_changed)

        # Add to main viewer
        win.add_view(nav_widget, location="top", orientation="horizontal")
        logger.debug("Navigation widget added")

    overall_elapsed = time.perf_counter() - overall_start
    # Restore window state (dock visibility, positions, sizes) after all viewers are added
    logger.debug("Restoring window state")
    win.restore_window_state()

    logger.info(f"Browser ready in {overall_elapsed:.3f}s")

    return win, dh5_file.name, cache


def main() -> None:
    """Main entry point for dh5browser command."""
    parser = argparse.ArgumentParser(
        description="Interactive browser for DH5 (DAQ-HDF5) files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dh5browser mydata.dh5
  dh5browser mydata.dh5 --trial 0
  dh5browser mydata.dh5 -t 2

The browser displays signals, spikes, events, and epochs from a single trial
using an interactive viewer based on ephyviewer.
        """,
    )

    parser.add_argument(
        "filename",
        type=str,
        nargs="?",
        help="Path to DH5 file to open (optional, will show file picker if not provided)",
    )
    parser.add_argument(
        "-t",
        "--trial",
        type=int,
        default=0,
        help="Trial index to display (default: 0)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10,
        help="Maximum number of trials to cache in memory (default: 10)",
    )
    parser.add_argument(
        "--prefetch-count",
        type=int,
        default=3,
        help="Number of trials to prefetch ahead in background (default: 3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging with detailed timing information",
    )

    args = parser.parse_args()

    # Configure logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    # Initialize app variable
    app = None

    # Get filename from argument or file picker
    if args.filename:
        dh5_file = pathlib.Path(args.filename)
    else:
        # Create Qt application early for file picker
        app = ephyviewer.mkQApp()

        # Show file picker dialog
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setWindowTitle("Select DH5 File")
        file_dialog.setNameFilter("DH5 Files (*.dh5);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                dh5_file = pathlib.Path(selected_files[0])
            else:
                print("No file selected. Exiting.")
                sys.exit(0)
        else:
            print("No file selected. Exiting.")
            sys.exit(0)

    # Validate file path
    if not dh5_file.exists():
        print(f"Error: File not found: {dh5_file}")
        sys.exit(1)

    if not dh5_file.suffix == ".dh5":
        print(f"Warning: File does not have .dh5 extension: {dh5_file}")

    # Create Qt application (if not already created for file picker)
    if app is None:
        app = ephyviewer.mkQApp()

    # Create and show browser window
    cache = None
    try:
        win, filename, cache = create_browser(
            dh5_file,
            trial_index=args.trial,
            cache_size=args.cache_size,
            prefetch_count=args.prefetch_count,
        )

        logger.info("Showing browser window...")
        show_start = time.perf_counter()
        win.show()
        logger.debug(f"win.show() took: {time.perf_counter() - show_start:.3f}s")

        print("\n" + "=" * 60)
        print("DH5 Browser")
        print("=" * 60)
        print(f"File: {dh5_file}")
        print(f"Trial: {args.trial}")
        print("=" * 60)
        print("\nBrowser window opened. Close window to exit.")

        # Run the Qt event loop
        app.exec()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error creating browser: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up cache and shutdown prefetch threads
        if cache is not None:
            logger.info("Shutting down prefetch threads...")
            cache.shutdown()
            logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
