import logging
import pathlib
import time
import typing
from dataclasses import dataclass

import h5py
import numpy
from neo.rawio.baserawio import (
    BaseRawIO,
    _event_channel_dtype,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
)

from dh5io.cont import Cont
from dh5io.dh5file import DH5File
from dh5io.trialmap import Trialmap

# Configure logging
logger = logging.getLogger(__name__)

# Global counter for profiling
_chunk_call_count = 0
_chunk_total_time = 0.0


@dataclass
class RawIOHeader:
    nb_block: int
    nb_segment: list[int] | None
    signal_streams: numpy.ndarray[typing.Any, numpy.dtype[_signal_stream_dtype]]
    signal_channels: numpy.ndarray[typing.Any, numpy.dtype[_signal_channel_dtype]]
    event_channels: numpy.ndarray[typing.Any, numpy.dtype[_event_channel_dtype]]
    spike_channels: numpy.ndarray[typing.Any, numpy.dtype[_spike_channel_dtype]]

    def __getitem__(self, item):
        return getattr(self, item)


class DH5RawIO(BaseRawIO):
    """
    Class for reading DAQ-HDF5 (*.dh5) files from the Kreiter lab.

    signal_stream : CONTn HDF5 group
    signal_channel : one column of CONTn/DATA array
    segment : trials in TRIALMAP
    block : dh5 file


    """

    rawmode: str = "one-file"
    filename: str | pathlib.Path
    _file: DH5File
    _trialmap: Trialmap | None
    header: RawIOHeader | None

    def __init__(self, filename: str | pathlib.Path):
        # raise exception if file does not exist
        if not pathlib.Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist")
        BaseRawIO.__init__(self)
        self.filename = filename
        self._file = DH5File(filename)
        self._trialmap = self._file.get_trialmap()
        self.header = None

    def __del__(self):
        if hasattr(self, "_file"):
            del self._file

    def _source_name(self) -> str | pathlib.Path:
        return self.filename

    def _parse_signal_channels(self) -> numpy.ndarray:
        """Read info about analog signal channels from DH5 file. Called by `_parse_header`"""
        signal_channels = []
        for cont in self._file.get_cont_groups():
            sampling_rate = 1.0 / (cont.sample_period / 1e9)
            all_calibrations = cont.calibration
            channels = cont.channels
            dtype = cont.data.dtype
            units = "V"
            offset = 0.0

            for channel_index in range(cont.n_channels):
                cont_name = cont.name if cont.name else f"CONT{cont.id}"
                channel_name: str = f"{cont_name}/{channel_index}"
                gain = (
                    all_calibrations[channel_index]
                    if all_calibrations is not None
                    else 1.0
                )
                signal_channels.append(
                    (
                        channel_name,
                        channel_name,  # currently identical to id
                        sampling_rate,
                        dtype,
                        units,
                        gain,
                        offset,
                        f"CONT{cont.id}",
                        "0",  # buffer_id
                    )
                )

        return numpy.array(signal_channels, dtype=_signal_channel_dtype)

    def _parse_spike_channels(self) -> numpy.ndarray:
        """Read info about spike channels from DH5 file. Called by `_parse_header`"""
        spike_channels = []
        waveform_units = "V"
        waveform_offset = 0.0

        for spike_group in self._file.get_spike_groups():
            spike_id = DH5File.get_spike_id_from_name(spike_group.name)
            unit_name = f"SPIKE{spike_id}/0"
            # TODO: loop over units in CLUSTER_INFO if present
            unit_id = f"#{spike_id}/0"

            waveform_gain = spike_group.attrs.get("Calibration")
            if waveform_gain is None:
                waveform_gain = 1.0
            elif isinstance(waveform_gain, numpy.ndarray):
                waveform_gain = waveform_gain[0]  # Use first channel calibration

            spike_params = spike_group.attrs.get("SpikeParams")
            waveform_left_samples = spike_params["preTrigSamples"]

            # sample period in DH5 is in nano seconds
            waveform_sampling_rate = 1 / (spike_group.attrs.get("SamplePeriod") / 1e9)
            spike_channels.append(
                (
                    unit_name,
                    unit_id,
                    waveform_units,
                    waveform_gain,
                    waveform_offset,
                    waveform_left_samples,
                    waveform_sampling_rate,
                )
            )
        return numpy.array(spike_channels, dtype=_spike_channel_dtype)

    def _parse_header(self):
        logger.debug("  _parse_header() called")
        parse_start = time.perf_counter()

        step = time.perf_counter()
        _trialmap = self._file.get_trialmap()
        logger.debug(f"    get_trialmap(): {time.perf_counter() - step:.3f}s")

        nb_segment = [1] if _trialmap is None else [int(len(_trialmap))]

        step = time.perf_counter()
        signal_streams = self._parse_signal_streams()
        logger.debug(f"    _parse_signal_streams(): {time.perf_counter() - step:.3f}s")

        step = time.perf_counter()
        signal_channels = self._parse_signal_channels()
        logger.debug(f"    _parse_signal_channels(): {time.perf_counter() - step:.3f}s")

        step = time.perf_counter()
        event_channels = self._parse_event_channels()
        logger.debug(f"    _parse_event_channels(): {time.perf_counter() - step:.3f}s")

        step = time.perf_counter()
        spike_channels = self._parse_spike_channels()
        logger.debug(f"    _parse_spike_channels(): {time.perf_counter() - step:.3f}s")

        self.header = RawIOHeader(
            nb_block=1,
            nb_segment=nb_segment,
            signal_streams=signal_streams,
            signal_channels=signal_channels,
            event_channels=event_channels,
            spike_channels=spike_channels,
        )

        step = time.perf_counter()
        self._generate_minimal_annotations()
        logger.debug(
            f"    _generate_minimal_annotations(): {time.perf_counter() - step:.3f}s"
        )

        # Add trialmap annotations to segments
        if self._trialmap is not None and len(self._trialmap) > 0:
            block_index = 0
            bl_ann = self.raw_annotations["blocks"][block_index]
            for seg_index in range(len(self._trialmap)):
                seg_ann = bl_ann["segments"][seg_index]
                trial = self._trialmap[seg_index]
                seg_ann["trial_no"] = int(trial["TrialNo"])
                seg_ann["stim_no"] = int(trial["StimNo"])
                seg_ann["trial_type_no"] = int(trial["StimNo"])  # Alias for StimNo
                seg_ann["outcome"] = int(trial["Outcome"])

        logger.debug(
            f"  _parse_header() total: {time.perf_counter() - parse_start:.3f}s"
        )

        # Reset chunk call statistics
        global _chunk_call_count, _chunk_total_time
        _chunk_call_count = 0
        _chunk_total_time = 0.0

    def _parse_event_channels(
        self,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[_event_channel_dtype]]:
        # Start with trials and raw events
        event_channels_list = [
            ("trials", "TRIALMAP", "epoch"),
            ("events", "EV02", "event"),
        ]

        # Discover unique event codes by scanning all events
        events = self._file.get_events_array()
        if events is not None and len(events) > 0:
            event_codes = events["event"]
            # Get unique absolute values of event codes (positive codes only)
            unique_codes = sorted(set(abs(code) for code in event_codes if code != 0))

            # Create one epoch channel per event type
            for code in unique_codes:
                event_channels_list.append(
                    (f"event_epochs_{int(code)}", f"EV02_EPOCHS_{int(code)}", "epoch")
                )

        event_channels = numpy.array(event_channels_list, dtype=_event_channel_dtype)
        return event_channels

    def _parse_signal_streams(
        self,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[_signal_stream_dtype]]:
        """Read info about signal streams from DH5 file. Called by `_parse_header`

        One CONT group in the HDF5 file corresponds to one signal stream.

        """

        signal_streams = []
        for cont_name in self._file.get_cont_group_names():
            # signal_stream_dtype has fields: name, id, buffer_id
            signal_streams.append((cont_name, cont_name, "0"))
        return numpy.array(signal_streams, dtype=_signal_stream_dtype)

    def _segment_t_start(self, block_index: int, seg_index: int):
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        if len(self._trialmap) == 0:
            raise NotImplementedError("Data without trials is not yet supported")

        # Return absolute time from DH5 file (in seconds)
        return self._trialmap[seg_index]["StartTime"] / 1e9

    def _segment_t_stop(self, block_index: int, seg_index: int):
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        if len(self._trialmap) == 0:
            raise NotImplementedError("Data without trials is not yet supported")

        # Return absolute time from DH5 file (in seconds)
        return self._trialmap[seg_index]["EndTime"] / 1e9

    # signal and channel zone
    def _get_signal_size(
        self, block_index: int, seg_index: int, stream_index: int
    ) -> int:
        """
        Return the size of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        stream_id: str = self.header.signal_streams[stream_index]["id"]
        cont_id = int(stream_id.replace("CONT", ""))
        cont = self._file.get_cont_group_by_id(cont_id)

        # Get trial start and end times in nanoseconds
        trial_start_ns = self._trialmap[seg_index]["StartTime"]
        trial_end_ns = self._trialmap[seg_index]["EndTime"]

        # Find the data region and samples corresponding to this trial
        index = cont.index
        sample_period_ns = cont.sample_period

        # Find which index region(s) contain this trial
        region_starts = index["time"]

        # Calculate sample indices for trial boundaries
        size = 0
        for i, region in enumerate(index):
            region_start_time = region["time"]
            region_offset = region["offset"]

            # Calculate end time of this region
            if i < len(index) - 1:
                next_offset = index[i + 1]["offset"]
                region_n_samples = next_offset - region_offset
            else:
                region_n_samples = cont.n_samples - region_offset

            region_end_time = region_start_time + region_n_samples * sample_period_ns

            # Check if trial overlaps with this region
            if trial_end_ns > region_start_time and trial_start_ns < region_end_time:
                # Calculate overlap
                overlap_start = max(trial_start_ns, region_start_time)
                overlap_end = min(trial_end_ns, region_end_time)
                overlap_samples = int((overlap_end - overlap_start) / sample_period_ns)
                size += overlap_samples

        return size

    def _get_signal_t_start(
        self, block_index: int, seg_index: int, stream_index: int
    ) -> float:
        """
        Return the t_start of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        # Return absolute trial start time from DH5 file (in seconds)
        return self._trialmap[seg_index]["StartTime"] / 1e9

    def _get_analogsignal_chunk(
        self,
        block_index: int,
        seg_index: int,
        i_start: int,
        i_stop: int,
        stream_index: int,
        channel_indexes: None | list[int] | numpy.ndarray,
    ) -> numpy.ndarray:
        """
        Return the samples from a set of AnalogSignals indexed
        by stream_index and channel_indexes (local index inner stream).

        RETURNS
        -------
            array of samples, with each requested channel in a column
        """
        global _chunk_call_count, _chunk_total_time
        _chunk_call_count += 1
        func_start = time.perf_counter()
        logger.debug(
            f"    _get_analogsignal_chunk called (call #{_chunk_call_count}, stream={stream_index}, seg={seg_index})"
        )

        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        stream_id: str = self.header.signal_streams[stream_index]["id"]
        cont_id = int(stream_id.replace("CONT", ""))

        step = time.perf_counter()
        cont = self._file.get_cont_group_by_id(cont_id)
        logger.debug(
            f"      get_cont_group_by_id({cont_id}): {time.perf_counter() - step:.3f}s"
        )

        if channel_indexes is None:
            channel_indexes = numpy.arange(cont.n_channels)

        # Get trial boundaries
        trial_start_ns = self._trialmap[seg_index]["StartTime"]
        trial_end_ns = self._trialmap[seg_index]["EndTime"]

        # Get index and data
        step = time.perf_counter()
        index = cont.index
        logger.debug(f"      cont.index access: {time.perf_counter() - step:.3f}s")

        step = time.perf_counter()
        data = cont.data
        logger.debug(f"      cont.data access: {time.perf_counter() - step:.3f}s")

        sample_period_ns = cont.sample_period

        # Find data samples corresponding to the trial
        step = time.perf_counter()
        all_samples = []

        for i, region in enumerate(index):
            region_start_time = region["time"]
            region_offset = region["offset"]

            # Calculate end of this region
            if i < len(index) - 1:
                next_offset = index[i + 1]["offset"]
                region_n_samples = next_offset - region_offset
            else:
                region_n_samples = cont.n_samples - region_offset

            region_end_time = region_start_time + region_n_samples * sample_period_ns

            # Check if trial overlaps with this region
            if trial_end_ns > region_start_time and trial_start_ns < region_end_time:
                # Calculate sample indices within this region
                if trial_start_ns <= region_start_time:
                    start_sample_in_region = 0
                else:
                    start_sample_in_region = int(
                        (trial_start_ns - region_start_time) / sample_period_ns
                    )

                if trial_end_ns >= region_end_time:
                    end_sample_in_region = region_n_samples
                else:
                    end_sample_in_region = int(
                        (trial_end_ns - region_start_time) / sample_period_ns
                    )

                # Get absolute indices in data array
                abs_start = region_offset + start_sample_in_region
                abs_end = region_offset + end_sample_in_region

                # Extract samples for this region
                region_samples = data[abs_start:abs_end, :]
                all_samples.append(region_samples)

        logger.debug(
            f"      index loop & data extraction: {time.perf_counter() - step:.3f}s"
        )

        # Concatenate all samples
        if not all_samples:
            return numpy.zeros((0, len(channel_indexes)), dtype=data.dtype)

        step = time.perf_counter()
        full_signal = numpy.concatenate(all_samples, axis=0)
        logger.debug(f"      concatenate samples: {time.perf_counter() - step:.3f}s")

        # Apply i_start and i_stop within the trial
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = full_signal.shape[0]

        result = full_signal[i_start:i_stop, channel_indexes]

        elapsed = time.perf_counter() - func_start
        _chunk_total_time += elapsed
        logger.debug(
            f"    _get_analogsignal_chunk(stream={stream_index}, seg={seg_index}) total: {elapsed:.3f}s (cumulative: {_chunk_total_time:.3f}s)"
        )

        return result

    # spiketrain and unit zone
    def _spike_count(
        self, block_index: int, seg_index: int, spike_channel_index
    ) -> int:
        """Return the number of spikes in a segment for a spike channel."""
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        spike_channel = self.header.spike_channels[spike_channel_index]
        spike_id = int(spike_channel["id"].split("/")[0].lstrip("#"))
        spike_group = self._file.get_spike_group_by_id(spike_id)

        if spike_group is None:
            return 0

        # Get trial boundaries (absolute times in DH5 file)
        trial_start_ns = self._trialmap[seg_index]["StartTime"]
        trial_end_ns = self._trialmap[seg_index]["EndTime"]

        # Get spike timestamps (absolute times in DH5 file)
        index = spike_group["INDEX"][()]

        # Count spikes within trial boundaries
        mask = (index >= trial_start_ns) & (index < trial_end_ns)
        return int(numpy.sum(mask))

    def _get_spike_timestamps(
        self,
        block_index: int,
        seg_index: int,
        spike_channel_index,
        t_start: float | None,
        t_stop: float | None,
    ):
        """Return spike timestamps for a segment (absolute times in seconds)."""
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        spike_channel = self.header.spike_channels[spike_channel_index]
        spike_id = int(spike_channel["id"].split("/")[0].lstrip("#"))
        spike_group = self._file.get_spike_group_by_id(spike_id)

        if spike_group is None:
            return numpy.array([], dtype=numpy.int64)

        # Get trial boundaries in nanoseconds (absolute times in DH5 file)
        trial_start_ns = self._trialmap[seg_index]["StartTime"]
        trial_end_ns = self._trialmap[seg_index]["EndTime"]

        # Get all spike timestamps (absolute times in DH5 file)
        index = spike_group["INDEX"][()]

        # Filter to trial boundaries
        mask = (index >= trial_start_ns) & (index < trial_end_ns)
        spike_times = index[mask]

        # Apply additional t_start/t_stop if provided (absolute times in seconds)
        if t_start is not None or t_stop is not None:
            if t_start is not None:
                abs_t_start_ns = t_start * 1e9
                spike_times = spike_times[spike_times >= abs_t_start_ns]
            if t_stop is not None:
                abs_t_stop_ns = t_stop * 1e9
                spike_times = spike_times[spike_times < abs_t_stop_ns]

        return spike_times

    def _rescale_spike_timestamp(
        self, spike_timestamps: numpy.ndarray, dtype: numpy.dtype
    ) -> numpy.ndarray:
        """Rescale spike timestamps from integer nanoseconds to float seconds."""
        # Convert from nanoseconds to seconds
        return spike_timestamps.astype(dtype) / 1e9

    def _get_spike_raw_waveforms(
        self,
        block_index: int,
        seg_index: int,
        spike_channel_index,
        t_start: float | None,
        t_stop: float | None,
    ) -> numpy.ndarray:
        """Return spike waveforms as a 3D array (nb_spike, nb_channel, nb_sample)."""
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        spike_channel = self.header.spike_channels[spike_channel_index]
        spike_id = int(spike_channel["id"].split("/")[0].lstrip("#"))
        spike_group = self._file.get_spike_group_by_id(spike_id)

        if spike_group is None:
            return numpy.zeros((0, 1, 0), dtype=numpy.int16)

        # Get spike parameters
        spike_params = spike_group.attrs.get("SpikeParams")
        samples_per_spike = spike_params["spikeSamples"]

        # Get trial boundaries (absolute times in DH5 file)
        trial_start_ns = self._trialmap[seg_index]["StartTime"]
        trial_end_ns = self._trialmap[seg_index]["EndTime"]

        # Get spike timestamps (absolute times in DH5 file) and filter to trial
        index = spike_group["INDEX"][()]
        mask = (index >= trial_start_ns) & (index < trial_end_ns)

        # Apply additional t_start/t_stop if provided (absolute times in seconds)
        if t_start is not None or t_stop is not None:
            if t_start is not None:
                abs_t_start_ns = t_start * 1e9
                mask = mask & (index >= abs_t_start_ns)
            if t_stop is not None:
                abs_t_stop_ns = t_stop * 1e9
                mask = mask & (index < abs_t_stop_ns)

        spike_indices = numpy.where(mask)[0]
        n_spikes = len(spike_indices)

        # Get waveform data
        data = spike_group["DATA"][()]
        n_channels = data.shape[1] if data.ndim > 1 else 1

        if n_spikes == 0:
            return numpy.zeros((0, n_channels, samples_per_spike), dtype=numpy.int16)

        # Extract waveforms
        waveforms = numpy.zeros(
            (n_spikes, n_channels, samples_per_spike), dtype=numpy.int16
        )

        for i, spike_idx in enumerate(spike_indices):
            start_sample = spike_idx * samples_per_spike
            end_sample = start_sample + samples_per_spike

            if data.ndim == 1:
                waveforms[i, 0, :] = data[start_sample:end_sample]
            else:
                waveforms[i, :, :] = data[start_sample:end_sample, :].T

        return waveforms

    def _event_count(self, block_index: int, seg_index: int, event_channel_index):
        """Return the number of events in a segment."""
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        event_channel = self.header.event_channels[event_channel_index]
        event_channel_name = event_channel["name"]
        if isinstance(event_channel_name, bytes):
            event_channel_name = event_channel_name.decode("utf-8")
        event_type = event_channel["type"]

        if event_type == b"epoch":
            # Check if this is the trials/trialmap channel
            if event_channel_name == "trials":
                return 1
            # Otherwise it's an event_epochs_N channel - count paired epochs for this event type
            elif event_channel_name.startswith("event_epochs_"):
                event_code = int(event_channel_name.split("_")[-1])
                timestamps, durations, labels = self._get_event_epochs(
                    seg_index, event_code
                )
                return len(timestamps)
            return 1
        elif event_type == b"event":
            # This is the EV02 event channel
            events = self._file.get_events_array()
            if events is None:
                return 0

            # Get trial boundaries
            trial_start_ns = self._trialmap[seg_index]["StartTime"]
            trial_end_ns = self._trialmap[seg_index]["EndTime"]

            # Count events within trial
            event_times = events["time"]
            mask = (event_times >= trial_start_ns) & (event_times < trial_end_ns)
            return int(numpy.sum(mask))

        return 0

    def _get_event_timestamps(
        self,
        block_index: int,
        seg_index: int,
        event_channel_index,
        t_start: float | None,
        t_stop: float | None,
    ):
        """Return event timestamps, labels, and durations for a segment."""
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        event_channel = self.header.event_channels[event_channel_index]
        event_channel_name = event_channel["name"]
        if isinstance(event_channel_name, bytes):
            event_channel_name = event_channel_name.decode("utf-8")
        event_id = event_channel["id"]
        event_type = event_channel["type"]

        if event_type == b"epoch":
            if event_channel_name.startswith("event_epochs_"):
                # Return onset/offset paired event epochs for this event type
                event_code = int(event_channel_name.split("_")[-1])
                return self._get_event_epochs(seg_index, event_code)
            else:
                # Return trial epoch information
                trial = self._trialmap[seg_index]
                trial_start_ns = trial["StartTime"]
                trial_end_ns = trial["EndTime"]
                duration_ns = trial_end_ns - trial_start_ns

                # Epoch uses absolute time from DH5 file
                timestamps = numpy.array([trial_start_ns], dtype=numpy.int64)
                labels = numpy.array(
                    [f"Trial_{trial['TrialNo']}_Stim_{trial['StimNo']}"], dtype="U"
                )
                durations = numpy.array([duration_ns], dtype=numpy.int64)

                return timestamps, durations, labels

        elif event_type == b"event":
            # Return EV02 events
            events = self._file.get_events_array()
            if events is None:
                return (
                    numpy.array([], dtype=numpy.int64),
                    numpy.array([], dtype=numpy.int64),
                    numpy.array([], dtype="U"),
                )

            # Get trial boundaries (absolute times in DH5 file)
            trial_start_ns = self._trialmap[seg_index]["StartTime"]
            trial_end_ns = self._trialmap[seg_index]["EndTime"]

            # Filter events to trial (using absolute times)
            event_times = events["time"]
            event_codes = events["event"]
            mask = (event_times >= trial_start_ns) & (event_times < trial_end_ns)

            timestamps = event_times[mask]

            labels = numpy.array(
                [f"Event_{code}" for code in event_codes[mask]], dtype="U"
            )
            durations = numpy.zeros(len(timestamps), dtype=numpy.int64)

            # Apply t_start/t_stop if provided (absolute times in seconds)
            if t_start is not None or t_stop is not None:
                if t_start is not None:
                    abs_t_start_ns = t_start * 1e9
                    keep = timestamps >= abs_t_start_ns
                    timestamps = timestamps[keep]
                    labels = labels[keep]
                    durations = durations[keep]
                if t_stop is not None:
                    abs_t_stop_ns = t_stop * 1e9
                    keep = timestamps < abs_t_stop_ns
                    timestamps = timestamps[keep]
                    labels = labels[keep]
                    durations = durations[keep]

            return timestamps, durations, labels

        return (
            numpy.array([], dtype=numpy.int64),
            numpy.array([], dtype=numpy.int64),
            numpy.array([], dtype="U"),
        )

    def _get_event_epochs(self, seg_index: int, event_code: int):
        """
        Pair onset/offset events into epochs for a specific event type.

        Positive event codes (1, 2, 3) represent onset (trigger ON).
        Negative event codes (-1, -2, -3) represent offset (trigger OFF).
        Returns epochs from onset to offset with true durations for the specified event code.

        Parameters
        ----------
        seg_index : int
            Trial/segment index
        event_code : int
            The event code to pair (positive value, e.g., 1, 2, 3)

        Returns
        -------
        timestamps : ndarray
            Onset times in nanoseconds
        durations : ndarray
            Durations in nanoseconds
        labels : ndarray
            Labels for each epoch
        """
        # Get raw events for this trial
        events = self._file.get_events_array()
        if events is None:
            return (
                numpy.array([], dtype=numpy.int64),
                numpy.array([], dtype=numpy.int64),
                numpy.array([], dtype="U"),
            )

        # Get trial boundaries
        trial_start_ns = self._trialmap[seg_index]["StartTime"]
        trial_end_ns = self._trialmap[seg_index]["EndTime"]

        # Filter events to trial
        event_times = events["time"]
        event_codes = events["event"]
        mask = (event_times >= trial_start_ns) & (event_times < trial_end_ns)

        trial_times = event_times[mask]
        trial_codes = event_codes[mask]

        if len(trial_times) == 0:
            return (
                numpy.array([], dtype=numpy.int64),
                numpy.array([], dtype=numpy.int64),
                numpy.array([], dtype="U"),
            )

        # Filter to only this event type (onsets and offsets)
        onsets = []
        offsets = []

        for t, code in zip(trial_times, trial_codes):
            if abs(code) == event_code:
                if code > 0:
                    onsets.append(t)
                elif code < 0:
                    offsets.append(t)

        onsets = sorted(onsets)
        offsets = sorted(offsets)

        # Pair onsets with offsets
        timestamps = []
        durations = []
        labels = []

        if not onsets:
            return (
                numpy.array([], dtype=numpy.int64),
                numpy.array([], dtype=numpy.int64),
                numpy.array([], dtype="U"),
            )

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
                    timestamps.append(onset_time)
                    durations.append(duration)
                    labels.append(f"Event_{int(event_code)}")

                offset_idx += 1
            else:
                # No matching offset - use default 100ms duration
                timestamps.append(onset_time)
                durations.append(int(0.1 * 1e9))  # 100ms in nanoseconds
                labels.append(f"Event_{int(event_code)}")

            onset_idx += 1

        if len(timestamps) == 0:
            return (
                numpy.array([], dtype=numpy.int64),
                numpy.array([], dtype=numpy.int64),
                numpy.array([], dtype="U"),
            )

        return (
            numpy.array(timestamps, dtype=numpy.int64),
            numpy.array(durations, dtype=numpy.int64),
            numpy.array(labels, dtype="U"),
        )

    def _rescale_event_timestamp(
        self, event_timestamps, dtype, event_channel_index=None
    ):
        """Rescale event timestamps from integer nanoseconds to float seconds."""
        return event_timestamps.astype(dtype) / 1e9

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index=None):
        """Rescale epoch durations from integer nanoseconds to float seconds."""
        return raw_duration.astype(dtype) / 1e9
