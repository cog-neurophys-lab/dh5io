import typing
import pathlib
import numpy
import numpy.typing as npt
import neo
from dh5io.dh5file import DH5File, _trialmap_dtype
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
    _trialmap: numpy.ndarray[typing.Any, numpy.dtype[_trialmap_dtype]] | h5py.Dataset | None
    header: RawIOHeader | None

    def __init__(self, filename: str | pathlib.Path):
        BaseRawIO.__init__(self)
        self.filename = filename
        self._file = DH5File(filename)
        self._trialmap = self._file.get_trialmap()
        self.header = None

    def __del__(self):
        del self._file

    def _source_name(self) -> str | pathlib.Path:
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
                cont_name = cont.name.removeprefix("/")
                channel_name: str = f"{cont_name}/{channel_index}"
                gain = all_calibrations[channel_index] if all_calibrations is not None else 1.0
                signal_channels.append(
                    (
                        channel_name,
                        channel_name,  # currently identical to id
                        sampling_rate,
                        dtype,
                        units,
                        gain,
                        offset,
                        cont_name.removeprefix("/"),
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
            event_channels=self._parse_event_channels(),
            spike_channels=self._parse_spike_channels(),
        )

        self._generate_minimal_annotations()

    def _parse_event_channels(
        self,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[_event_channel_dtype]]:

        event_channels = numpy.array(
            [("trials", "TRIALMAP", "epoch"), ("events", "EV02", "event")],
            dtype=_event_channel_dtype,
        )
        return event_channels

    def _parse_signal_streams(
        self,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[_signal_stream_dtype]]:
        """Read info about signal streams from DH5 file. Called by `_parse_header`

        One CONT group in the HDF5 file corresponds to one signal stream.

        """

        signal_streams = []
        for cont_name in self._file.get_cont_group_names():
            signal_streams.append((cont_name, cont_name))
        return numpy.array(signal_streams, dtype=_signal_stream_dtype)

    def _segment_t_start(self, block_index: int, seg_index: int):
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self.header.nb_segment == 1:
            raise NotImplementedError("Data without trials is not yet supported")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        return self._trialmap[seg_index]["StartTime"]

    def _segment_t_stop(self, block_index: int, seg_index: int):
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self.header.nb_segment == 1:
            raise NotImplementedError("Data without trials is not yet supported")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        return self._trialmap[seg_index]["EndTime"]

    # signal and channel zone
    def _get_signal_size(self, block_index: int, seg_index: int, stream_index: int) -> int:
        """
        Return the size of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        if self.header is None:
            raise ValueError("Header not yet parsed")

        if self._trialmap is None:
            raise ValueError("Trialmap not yet parsed")

        contId: str = self.header.signal_streams[stream_index]["id"]
        data: h5py.Dataset = self._file.file[contId]["DATA"]
        index: h5py.Dataset = self._file.file[contId]["INDEX"]

        # FIXME: clarify how a neo segment maps to a trial / an area within a CONT block
        # Segments are the trials in the trialmap. We need to find the indices in the data array
        # that correspond to the start and end of the trial.
        # index contains the start time and the offset in the data array, i.e. we can
        # construct the time axis based on this information.

        iStart: int = index[seg_index]["offset"]

        if seg_index == len(self._trialmap):
            iEnd: int = data.shape[0]
        else:
            iEnd = index[seg_index + 1]["offset"]

        return iEnd - iStart

    def _get_signal_t_start(self, block_index: int, seg_index: int, stream_index: int) -> float:
        """
        Return the t_start of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        if self.header is None:
            raise ValueError("Header not yet parsed")

        contId: str = self.header.signal_streams[stream_index]["id"]
        index: h5py.Dataset = self._file.file[contId]["INDEX"]
        return index[seg_index]["time"] / 1e9

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
        if self.header is None:
            raise ValueError("Header not yet parsed")

        contId: str = self.header.signal_streams[stream_index]["id"]

        if channel_indexes is None:
            channel_indexes = numpy.arange(self._file.file[contId]["DATA"].shape[1])

        return numpy.array(self._file.file[contId][i_start:i_stop, channel_indexes])

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
