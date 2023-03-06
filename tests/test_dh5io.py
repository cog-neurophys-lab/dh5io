import pytest
import dh5io
import pathlib

filename = pathlib.Path(__file__).parent / "test.dh5"


@pytest.fixture
def test_file() -> dh5io.DH5RawIO:
    return dh5io.DH5RawIO(filename)


class TestDH5RawIO:
    def test_load(self, test_file: dh5io.DH5RawIO):
        test_file.parse_header()

        assert test_file.signal_streams_count() == 7
        assert test_file.signal_channels_count(0) == 1
        raw_chunk = test_file.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=1024, channel_indexes=None,
            stream_index=1
        )
        assert raw_chunk.shape == (1024,1)


# >>> import neo.rawio
# >>> r = neo.rawio.ExampleRawIO(filename='itisafake.nof')
# >>> r.parse_header()
# >>> print(r)
# >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
#                     i_start=0, i_stop=1024,  channel_names=channel_names)
# >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
#                     channel_indexes=[0, 3, 6])
# >>> spike_timestamp = reader.spike_timestamps(spike_channel_index=0,
#                     t_start=None, t_stop=None)
# >>> spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
# >>> ev_timestamps, _, ev_labels = reader.event_timestamps(event_channel_index=0)

# class TestDH5IO:

# def test_load(self):
# dh5 = dh5io.DH5IO(filename)
