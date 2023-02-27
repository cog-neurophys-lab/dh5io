import pathlib
import neo
from neo.rawio.baserawio import BaseRawIO


class DH5RawIO(BaseRawIO):
    """
    Class for reading DAQ-HDF5 (*.dh5) files from the Kreiter lab.
    """

    rawmode = "one-file"

    def __init__(self, filename: str | pathlib.Path):
        BaseRawIO.__init__(self)
        # note that this filename is ued in self._source_name
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        raise (NotImplementedError)
        # must call
        # self._generate_empty_annotations()
        """ This must create
        self.header['nb_block']
        self.header['nb_segment']
        self.header['signal_streams']
        self.header['signal_channels']
        self.header['spike_channels']
        self.header['event_channels']"""
        self.header = {}
        


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
