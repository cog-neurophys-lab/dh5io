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
class header:
    nb_block: int
    nb_segment: int
    signal_streams: numpy.ndarray  # with dtype=_signal_stream_dtype


_trialmap_dtype = [
    ("name", "U64"),  # not necessarily unique
    ("id", "U64"),  # must be unique
    ("sampling_rate", "float64"),
    ("dtype", "U16"),
    ("units", "U64"),
    ("gain", "float64"),
    ("offset", "float64"),
    ("stream_id", "U64"),
]


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
        return int(name.lstrip('/').lstrip("CONT"))


class DH5RawIO(BaseRawIO):
    """
    Class for reading DAQ-HDF5 (*.dh5) files from the Kreiter lab.
    """

    rawmode: str = "one-file"
    filename: str | pathlib.Path
    _file: DH5File

    def __init__(self, filename: str | pathlib.Path):
        BaseRawIO.__init__(self)
        self.filename = filename
        self._file = DH5File(filename)

    def __del__(self):
        del self._file

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        """This must create
        self.header["nb_block"]
        self.header["nb_segment"]
        self.header["signal_streams"]
        self.header["signal_channels"]
        self.header["spike_channels"]
        self.header["event_channels"]"""
        
        self.header = {}
        self.header["nb_block"] = 1
        trialmap = self._file.get_trialmap()
        self.header["nb_segment"] = 1 if trialmap is None else trialmap.size

        signal_streams = []
        for cont_name in self._file.get_cont_group_names():
            stream_id = DH5File.get_cont_id_from_name(cont_name)
            signal_streams.append((cont_name, stream_id))
        signal_streams = numpy.array(signal_streams, dtype=_signal_stream_dtype)
        self.header["signal_streams"] = signal_streams

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
                gain = (
                    all_calibrations[channel_index]
                    if all_calibrations is not None
                    else 1.0
                )
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

        signal_channels = numpy.array(signal_channels, dtype=_signal_channel_dtype)
        self.header["signal_channels"] = signal_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        raise (NotImplementedError)

    def _segment_t_stop(self, block_index, seg_index):
        raise (NotImplementedError)

    ###
    # signal and channel zone
    def _get_signal_size(self, block_index, seg_index, stream_index):
        """
        Return the size of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        raise (NotImplementedError)

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        """
        Return the t_start of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        raise (NotImplementedError)

    def _get_analogsignal_chunk(
        self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes
    ):
        """
        Return the samples from a set of AnalogSignals indexed
        by stream_index and channel_indexes (local index inner stream).

        RETURNS
        -------
            array of samples, with each requested channel in a column
        """

        raise (NotImplementedError)

    # spiketrain and unit zone
    def _spike_count(self, block_index, seg_index, spike_channel_index):
        raise (NotImplementedError)

    def _get_spike_timestamps(
        self, block_index, seg_index, spike_channel_index, t_start, t_stop
    ):
        raise (NotImplementedError)

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        raise (NotImplementedError)

    # spike waveforms zone
    def _get_spike_raw_waveforms(
        self, block_index, seg_index, spike_channel_index, t_start, t_stop
    ):
        raise (NotImplementedError)

    ###
    # event and epoch zone
    def _event_count(self, block_index, seg_index, event_channel_index):
        raise (NotImplementedError)

    def _get_event_timestamps(
        self, block_index, seg_index, event_channel_index, t_start, t_stop
    ):
        raise (NotImplementedError)

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        raise (NotImplementedError)

    def _rescale_epoch_duration(self, raw_duration, dtype):
        raise (NotImplementedError)
