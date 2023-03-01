import pathlib
import numpy
import neo
from neo.rawio.baserawio import BaseRawIO
import h5py
from dataclasses import dataclass
from neo.rawio.baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)


@dataclass
class RawIOHeader:
    nb_block: int
    nb_segment: list[int]
    signal_streams: "numpy.ndarray[_signal_stream_dtype]"
    signal_channels: "numpy.ndarray[_signal_channel_dtype]"
    event_channels: "numpy.ndarray[_event_channel_dtype]"
    spike_channels: "numpy.ndarray[_spike_channel_dtype]"

    def __getitem__(self, item):
        return getattr(self, item)


class DH5File:
    file: h5py.File

    def __init__(self, filename: str | pathlib.Path, mode="r"):
        self.file = h5py.File(filename, mode)

    def get_version(self) -> int | None:
        return self.file.attrs.get("version")

    def get_cont_groups(self) -> list[h5py.Group]:
        return [self.file[name] for name in self.get_cont_group_names()]

    def get_cont_group_names(self) -> list[str]:
        return [
            name
            for name in self.file.keys()
            if name.startswith("CONT") and isinstance(self.file[name], h5py.Group)
        ]

    def get_cont_group_by_id(self, id: int) -> h5py.Group | None:
        return self.file.get(f"CONT{id}")

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

    def get_cont_data_by_id(self, cont_id) -> h5py.Dataset:
        return self.get_cont_group_by_id(cont_id).get("DATA")

    def get_cont_size(self, cont_id) -> tuple[int, int]:
        nSamples, nChannels = self.get_cont_data_by_id(cont_id).shape
        return (nSamples, nChannels)

    def get_trialmap(self) -> h5py.Dataset | None:
        return self.file.get("TRIALMAP")

    def __del__(self):
        self.file.close()

    @staticmethod
    def get_cont_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("CONT"))

    @staticmethod
    def get_spike_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("SPIKE"))


class DH5RawIO(BaseRawIO):
    """
    Class for reading DAQ-HDF5 (*.dh5) files from the Kreiter lab.
    """

    rawmode: str = "one-file"
    filename: str | pathlib.Path
    _file: DH5File
    _trialmap: h5py.Dataset | None
    header: RawIOHeader

    def __init__(self, filename: str | pathlib.Path):
        BaseRawIO.__init__(self)
        self.filename = filename
        self._file = DH5File(filename)
        self._trialmap = None

    def __del__(self):
        del self._file

    def _source_name(self):
        return self.filename

    def _parse_signal_channels(self) -> numpy.ndarray:
        """Read info about analog signal channels from DH5 file. Called by `_parse_header`"""
        signal_channels = []
        for cont in self._file.get_cont_groups():
            data: h5py.Dataset = cont["DATA"]
            index: h5py.Dataset = cont["INDEX"]

            sampling_rate = 1.0 / (cont.attrs["SamplePeriod"] / 1e9)
            all_calibrations = cont.attrs.get("Calibration")
            channels = cont.attrs.get("Channels")
            dtype = data.dtype
            units = "V"
            offset = 0.0

            for channel_index in range(data.shape[1]):
                channel_name = f"{cont.name}/{channel_index}"

                channel_id = (
                    channels["GlobalChanNumber"][channel_index]
                    if channels is not None
                    else 1000 * DH5File.get_cont_id_from_name(cont.name.strip("/"))
                    + channel_index
                )
                gain = all_calibrations[channel_index] if all_calibrations is not None else 1.0
                signal_channels.append(
                    (
                        channel_name,
                        channel_id,
                        sampling_rate,
                        dtype,
                        units,
                        gain,
                        offset,
                        DH5File.get_cont_id_from_name(cont.name),
                    )
                )

        return numpy.array(signal_channels, dtype=_signal_channel_dtype)

    def _parse_spike_channels(self) -> numpy.ndarray:
        """Read info about spike channels from DH5 file. Called by `_parse_header`"""
        spike_channels = []
        waveform_units = "V"
        waveform_offset = 0.0

        for spike_group in self._file.get_spike_groups():
            unit_name = f"{spike_group.name}/0"  # "unit{}".format(c)
            # TODO: loop over units in CLUSTER_INFO if present
            unit_id = f"#{DH5File.get_spike_id_from_name(spike_group.name)}/0"

            waveform_gain = spike_group.attrs.get("Calibration")
            if waveform_gain is None:
                waveform_gain = 1.0

            waveform_left_samples = spike_group.attrs.get("SpikeParams")["preTrigSamples"]

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
        _trialmap = self._file.get_trialmap()
        nb_segment = [1] if _trialmap is None else [int(_trialmap.size)]

        self.header = RawIOHeader(
            nb_block=1,
            nb_segment=nb_segment,
            signal_streams=self._parse_signal_streams(),
            signal_channels=self._parse_signal_channels(),
            event_channels=self._parse_spike_channels(),
            spike_channels=self._parse_event_channels(),
        )

        self._generate_minimal_annotations()

    def _parse_event_channels(self):
        event_channels = []
        event_channels.append(("trials", "TRIALMAP", "epoch"))
        event_channels.append(("events", "EV02", "event"))
        event_channels = numpy.array(event_channels, dtype=_event_channel_dtype)
        return event_channels

    def _parse_signal_streams(self):
        """Read info about spike channels from DH5 file. Called by `_parse_header`"""

        signal_streams = []
        for cont_name in self._file.get_cont_group_names():
            stream_id = DH5File.get_cont_id_from_name(cont_name)
            signal_streams.append((cont_name, stream_id))
        signal_streams = numpy.array(signal_streams, dtype=_signal_stream_dtype)
        return signal_streams

    def _segment_t_start(self, block_index: int, seg_index: int):
        if self.header.nb_segment > 1:
            return self._trialmap[seg_index]["StartTime"]
        else:
            NotImplementedError("Data without trials is not yet supported")

    def _segment_t_stop(self, block_index: int, seg_index: int):
        if self.header.nb_segment > 1:
            return self._trialmap[seg_index]["EndTime"]
        else:
            NotImplementedError("Data without trials is not yet supported")

    # signal and channel zone
    def _get_signal_size(self, block_index: int, seg_index: int, stream_index: int) -> int:
        """
        Return the size of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        raise (NotImplementedError)

    def _get_signal_t_start(self, block_index: int, seg_index: int, stream_index: int):
        """
        Return the t_start of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        raise (NotImplementedError)

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

        # This must return a numpy array 2D (even with one channel).
        raise (NotImplementedError)

    # spiketrain and unit zone
    def _spike_count(self, block_index: int, seg_index: int, spike_channel_index) -> int:
        raise (NotImplementedError)

    def _get_spike_timestamps(
        self,
        block_index: int,
        seg_index: int,
        spike_channel_index,
        t_start: float | None,
        t_stop: float | None,
    ):
        raise (NotImplementedError)

    def _rescale_spike_timestamp(
        self, spike_timestamps: numpy.ndarray, dtype: numpy.dtype
    ) -> numpy.ndarray:
        raise (NotImplementedError)

    def _get_spike_raw_waveforms(
        self,
        block_index: int,
        seg_index: int,
        spike_channel_index,
        t_start: float | None,
        t_stop: float | None,
    ) -> numpy.ndarray:
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)

        raise (NotImplementedError)

    def _event_count(self, block_index: int, seg_index: int, event_channel_index):
        raise (NotImplementedError)

    def _get_event_timestamps(
        self,
        block_index: int,
        seg_index: int,
        event_channel_index,
        t_start: float | None,
        t_stop: float | None,
    ):
        raise (NotImplementedError)

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        raise (NotImplementedError)

    def _rescale_epoch_duration(self, raw_duration, dtype):
        raise (NotImplementedError)
