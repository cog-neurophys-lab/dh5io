"""Spike data in SPIKE blocks in DAQ-HDF files.

Brain signals recorded from selective electrodes contain spikes. Most
analysis algorithms for such signals are interested in spikes only, not
the signal waveform between them. During the recording or after the
recording, in an off-line processing, spikes are detected and extracted
from continuous-time signals. For each spike, a piece of signal waveform
that contains the spike, is stored as well as a timestamp. Spikes are
usually sorted in the first stages of data processing where, depending
on their waveform, each of them is assigned to one of several spike
clusters. It is possible to save this clustering information in DAQ-HDF
files, too. Spike data is represented in DAQ-HDF files in the form of
`SPIKE` blocks.

`SPIKE` blocks also use the concept of nTrodes, like the `CONT` blocks.
Multichannel data is possible within a single `SPIKE` block, however, all
the channels have the same sampling rate, the same time windows for
stored waveforms, and the same spike timestamps.

SPIKE` blocks are stored in groups named `SPIKEn`, where n can have values
between 0 and 65535. `CONT` and `SPIKE` blocks can have the same
identifiers.

`SPIKEn` group **must** have the following **attributes**:

- `SpikeParams` (`struct` scalar):

    | Offset    | Name               | Type      |
    |-----|----------------|-------|
    | 0   | spikeSamples   | `int16` |
    | 2   | preTrigSamples | `int16` |
    | 4   | lockOutSamples | `int16` |

- `Channels` (`struct` array\[N\]), with the same members and their meaning as
the 'Channels' attribute for `CONTn` groups;

- `SamplePeriod` (`int32` scalar).

*Optional attribute*:

- `Calibration` (`double` array\[N\])

`SPIKEn` group **must** have the following **datasets**:

- `DATA` (`int16` array\[M,N\]);
- `INDEX` (`int64` array\[S\]);

*Optional dataset*:

- `CLUSTER_INFO` (unsigned `int8` array\[S\]).

Here, `N` is number of channels in nTrode; `M` is the total number of
samples stored for every channel in the `SPIKE` block; S is the total
number of spikes.

Description of the **attributes**:

- `Channels`, `SamplePeriod` and `Calibration` attributes play the same
role here as in the `CONT` blocks.

- `SpikeParams` describes some spike parameters common for all the
channels within this nTrode. `spikeSamples` member tells how many
samples of the signal waveform are stored in total for each spike;
`PreTrigSamples` member specifies how many samples of the signal
waveform are stored prior to the spike trigger point. `lockOutSamples`
is a parameter which was used for detection of spikes and tells the
minimum number of samples between the trigger points of two consecutive
spikes.

Description of the **datasets**:

- `DATA` stores all spike waveforms, merged together. Therefore the total
number of samples M is equal to the product of SpikeParams.spikeSamples
and the total number of spikes S. Waveforms are stored in 16-bit signed
format, same as with `CONT` blocks.

- `INDEX` stores spike timestamps, specified in nanoseconds. There are S
timestamps, one for each spike, and it tells the time of the spike
trigger point which is not the beginning of a particular waveform if
spikeParams.PreTrigSamples is nonzero.

Because the length of all spike waveforms is the same, it is simple to
extract waveform for a particular spike: sample offset is calculated by
multiplying the spike number with the `spikeParams.spikeSamples`
parameter.

If there are multiple channels in the nTrode, spike trigger points are
common for all these channels, as well as other parameters except the
waveforms themselves.

- `CLUSTER_INFO` dataset is created during the spike sorting process. Each
spike is assigned a cluster number, so the `CLUSTER_INFO` dataset simply
stores these numbers for every spike. There can be up to 256 clusters
for every `SPIKE` block, which is far more than enough, since a typical
spike sorting process creates 2 to 4 clusters.

"""

raise ImportError("nothing here yet")
