"""Signal data in CONT blocks of DAQ-HDF files.

Signal data for DAQ-HDF means continuously or piecewise-continuously
(trial-based) recorded continuous-time signal. Sampling is supposed to
be equidistant. It is possible to store multiple signals with different
sampling rates and different regions of recording. Signal data is
represented in DAQ-HDF files in the form of `CONT` blocks (from the
continuous-time signal concept).

Each `CONT` block stores data for a single nTrode which is a multi-channel
electrode. Therefore, a `CONT` block can contain multiple channels of
piecewise-continuous signal recording. All channels within a `CONT` block
share the same sampling rate and the same regions (pieces) of recording.
Different `CONT` blocks, however, can have both of the mentioned
parameters independent of each other. Each `CONT` block has an unique
identifying number, and apart from a range limit, there are no other
restrictions which Ids to assign to them. In contrast, channels within a
single `CONT` block are numbered from 0 to N-1, where N is the number of
channels. Gaps in numbering are not possible.

In general, `CONT` blocks are thought of to be separable units of data,
whereas the channels within a single `CONT` block are supposed be stored
and processed together.

There can be several `CONT` blocks in a DAQ-HDF file. Each of them is
stored in a group named `CONTn`, where n is the identifier number of each
`CONT` block. This identifier must be in the range from 0 to 65535.

`CONTn` group **must** have the following **attributes**:

- `Channels` (`struct` array[N]):

    | Offset | Name | Type      |
    |-----|------------------|-------|
    | 0   | GlobalChanNumber | `int16` |
    | 2   | BoardChanNo      | `int16` |
    | 4   | ADCBitWidth      | `int16` |
    | 6   | MaxVoltageRange  | `float` |
    | 10  | MinVoltageRange  | `float` |
    | 14  | AmplifChan0      | `float` |

- `SamplePeriod` (`int32` scalar).

*Optional attribute*:

- `Calibration` (`double` array[N])

`CONTn` group **must** have the following **datasets**:

- `DATA` (`int16` array[M,N])

- `INDEX` (`struct` array[R]):

    | Offset    | Name       | Type  |
    |-----|--------|-------|
    | 0   | time   | `int64` |
    | 8   | offset | `int64` |

Here, N is number of channels in nTrode; M is the total number of
samples stored for every channel in the `CONT` block; R is the number of
recording regions.

Shared HDF5 datatype `/CONT_INDEX_ITEM` is used in each `CONT` block to
describe the INDEX dataset.

Description of the **attributes**:

Signal data may be recorded from multiple A/D boards within a single PC.
Data Acquisition Program enumerates all available A/D channels from all
A/D boards present, so that each channel gets an unique number at the
time of recording. This is stored in the `GlobalChanNumber` member of the
structure. A/D channels which compose an nTrode may have very different
numbers, they may also belong to different A/D boards in the recording
setup. This information is normally not needed during the data
processing, but may be needed for documentation of the experiment.

- `BoardChanNo` - this is the number of channel within the A/D board from
which it was acquired.

- `ADCBitWidth` – number of bits in the A/D converter. Note, however, that
the signals are always stored in 16-bit format regardless of the value
of this parameter.

- `MaxVoltageRange`, `MinVoltageRange` – these two values specify the A/D
converter's input voltage range. Knowing them, it is possible to convert
the unitless signal data into volts.

- `AmplifChan0` – If an A/D board has some programmable-gain amplifier
(PGA), this value specifies amplification gain for each recording
channel. If this value is zero, then there is no PGA on the board.

- `SamplePeriod` is specified in nanoseconds. It's the time interval between
two consecutive samples of the signal.

- `Calibration` attribute stores a real number for every channel belonging
to the nTrode. If you multiply this calibration value with the raw
channel data, you get value in volts. `Calibration` attribute is normally
not present in a freshly recorded and converted file, because there is
not enough information to produce the calibration value. It must be
obtained from other source of information, typically these are
special-purpose calibration recording files.

`Calibration` value is supposed to encapsulate all the gains throughout
the whole amplification/recording chain. By multiplying calibration
value with channel data it should be possible, therefore, to get the
very initial voltage as it was on the electrode tip.

Description of the **datasets**:

- `DATA` dataset stores the signal samples as a single 2-dimensional block in the form of
16-bit integers, whose minimum value is -32768, and the maximum value is 32767. Contiguous
pieces of recording are merged together. It is possible to determine where these pieces
(regions) are located by using information from the INDEX dataset.

- `INDEX` structure dataset characterizes each recording region with two numbers: 'time' is
the timestamp of the first signal sample, in nanoseconds, and 'offset' member specifies the
sample offset within the `DATA` dataset where is the first sample of a particular region
stored.

Knowing these two values for each recording region, and knowing the
total number of samples, it is possible to calculate the following
information: offsets of starting and ending sample for each recording
region, and their respective time stamps.


"""

import logging
import warnings

import h5py
import numpy as np
import numpy.typing as npt

from dh5io.errors import (
    DH5Error,
    DH5Warning,
    DH5CalibrationMissingWarning,
    DH5ChannelsMissingWarning,
    DH5DataTypeConversionWarning,
)
from dhspec.cont import (
    CONT_DTYPE_NAME,
    CONT_PREFIX,
    DATA_DATASET_NAME,
    INDEX_DATASET_NAME,
    CalibrationType,
    ContSignalType,
    cont_id_from_name,
    cont_name_from_id,
    create_channel_info,
    create_empty_index_array,
)

logger = logging.getLogger(__name__)


class Cont:
    """Abstraction for the HDF5 Group containing continuous signal data"""

    def __init__(self, group: h5py.Group):
        self._group = group

    def __str__(self):
        # show a tree view with all the properties below
        return f"""    {self._group.name} in {self._group.file.filename}
        ├─── id: {self.id}
        ├─── name: {self.name}
        ├─── comment: {self.comment}
        ├─── sample_period: {self.sample_period} ns ({1 / self.sample_period * 1e9} Hz)
        ├─── n_channels: {self.n_channels}
        ├─── n_samples: {self.n_samples}
        ├─── duration: {self.duration_s:.2f} s
        ├─── n_regions: {self.n_regions}
        ├─── signal_type: {self.signal_type.name if self.signal_type else "None"}
        ├─── calibration: {self.calibration if self.calibration is not None else "None"}
        ├─── data: {self.data.shape}
        └─── index: {self.index.shape}
            """

    def __repr__(self):
        return f"Cont(group={self._group.name}, file={self._group.file.filename})"

    # properties

    @property
    def id(self) -> int:
        """Return the integer identifier of the continuous signal."""
        return cont_id_from_name(self._group.name)

    @property
    def data(self) -> npt.NDArray[np.int16]:
        """Return the raw integer signal data (nSamples, nChannels)."""
        return self._group[DATA_DATASET_NAME][()]

    @property
    def index(self) -> np.ndarray:
        """Return the index array (regions, fields: time, offset)."""
        return self._group[INDEX_DATASET_NAME][()]

    @property
    def calibration(self) -> CalibrationType | None:
        """Return the calibration array or None if missing."""
        return self._group.attrs.get("Calibration")

    @property
    def channels(self) -> np.ndarray | None:
        """Return the channels attribute array or None if missing."""
        return self._group.attrs.get("Channels")

    @property
    def sample_period(self) -> int:
        """Return the sample period in nanoseconds."""
        return int(self._group.attrs["SamplePeriod"])

    @property
    def name(self) -> str:
        """Return the name attribute."""
        return self._group.attrs.get("Name", "")

    @property
    def comment(self) -> str:
        """Return the comment attribute."""
        return self._group.attrs.get("Comment", "")

    @property
    def signal_type(self) -> ContSignalType | None:
        """Return the signal type as ContSignalType or None."""
        val = self._group.attrs.get("SignalType")
        if val is not None:
            try:
                return ContSignalType(val)
            except ValueError:
                return None
        return None

    @property
    def n_channels(self) -> int:
        """Return the number of channels in the CONT block."""
        return self._group[DATA_DATASET_NAME].shape[1]

    @property
    def n_samples(self) -> int:
        """Return the number of samples."""
        return self._group[DATA_DATASET_NAME].shape[0]

    @property
    def duration_s(self) -> float:
        """Return the total duration of the CONT block in seconds."""
        if self.n_regions == 0:
            return 0.0
        index = self.index
        sample_period = self.sample_period
        data_shape = self.data.shape
        if data_shape[0] == 0:
            return 0.0
        # Last region's offset and number of samples
        last_region = index[-1]
        last_offset = last_region["offset"]
        n_samples = data_shape[0]
        # Start time of first region
        start_time = index[0]["time"]
        # End time of last sample
        end_time = last_region["time"] + (n_samples - last_offset) * sample_period
        return (end_time - start_time) / 1e9

    @property
    def n_regions(self) -> int:
        """Return the number of regions (index items)."""
        return self.index.shape[0]

    @property
    def calibrated_data(self) -> npt.NDArray[np.float64]:
        """Return calibrated data if calibration is present, else raw data."""
        calib = self.calibration
        if calib is None:
            warnings.warn(
                DH5CalibrationMissingWarning(
                    f"Calibration attribute is missing from {self._group.name}",
                    cont_id=self.id,
                )
            )
            return self.data.astype(np.float64)
        return self.data * calib

    @classmethod
    def from_group(cls, group: h5py.Group) -> "Cont":
        """Create a Cont instance from an h5py.Group."""
        return cls(group)


# create
def create_empty_cont_group_in_file(
    file: h5py.File,
    cont_group_id: int | None,
    nSamples: int,
    nChannels: int,
    sample_period_ns: np.int32,
    n_index_items: int = 1,
    # numpy array with dtype=np.float64 of length nChannels describing calibration
    calibration: CalibrationType | None = None,
    # numpy array with dtype=CHANNELS_DTYPE of length nChannels describing channels
    channels: np.ndarray | None = None,
    name: str | None = None,
    comment: str | None = None,
    signal_type: ContSignalType | None = None,
) -> h5py.Group:
    existing_cont_ids = enumerate_cont_groups(file)

    # check if opened with write access
    if not file.mode == "r+" and not file.mode == "w" and not file.mode == "a":
        raise DH5Error(
            f"File must be opened with write access but is open with {file.mode}"
        )

    # fail if CONT group already exists
    if cont_group_id in existing_cont_ids:
        raise DH5Error(f"CONT{cont_group_id} already exists in {file.filename}")

    if cont_group_id is None:
        cont_group_id = np.max(np.array(existing_cont_ids)) + 1
        logger.debug(
            f"No CONT group id provided, creating new CONT group {cont_group_id}"
        )

    cont_group = file.create_group(cont_name_from_id(cont_group_id))

    cont_group.create_dataset(
        DATA_DATASET_NAME, shape=(nSamples, nChannels), dtype=np.int16
    )
    cont_group.create_dataset(
        INDEX_DATASET_NAME, shape=(n_index_items,), dtype=file[CONT_DTYPE_NAME]
    )

    cont_group.attrs["SamplePeriod"] = np.int32(sample_period_ns)

    # optional attributes
    if calibration is not None:
        cont_group.attrs["Calibration"] = calibration

    if channels is not None:
        cont_group.attrs["Channels"] = channels

    # set name attribute
    if name is not None:
        cont_group.attrs["Name"] = name
    else:
        cont_group.attrs["Name"] = f"CONT{cont_group_id}"
        logger.debug(
            f"Name attribute not provided, using default {cont_group.attrs['Name']}"
        )

    # set comment attribute
    if comment is not None:
        cont_group.attrs["Comment"] = comment
    else:
        cont_group.attrs["Comment"] = ""

    # set signal type attribute
    if signal_type is not None:
        cont_group.attrs["SignalType"] = signal_type.value

    return cont_group


def create_cont_group_from_data_in_file(
    file: h5py.File,
    cont_group_id: int,  # group name will be CONT_{cont_group_id}
    data: np.ndarray,
    index: np.ndarray,
    sample_period_ns: np.int32,
    calibration: CalibrationType | None = None,
    channels: np.ndarray | None = None,
    name: str | None = None,
    comment: str | None = None,
    signal_type: ContSignalType | None = None,
) -> h5py.Group:
    cont_group = create_empty_cont_group_in_file(
        file,
        cont_group_id,
        nSamples=data.shape[0],
        nChannels=data.shape[1],
        sample_period_ns=sample_period_ns,
        n_index_items=index.shape[0],
        calibration=calibration,
        channels=channels,
        name=name,
        comment=comment,
        signal_type=signal_type,
    )

    # make sure data in integer type
    if not data.dtype == np.int16:
        warnings.warn(
            DH5DataTypeConversionWarning(
                f"Data was converted from {data.dtype} to numpy.int16"
            )
        )
        data = data.astype(np.int16)
    cont_group["DATA"][:] = data
    cont_group["INDEX"][:] = index

    return cont_group


def enumerate_cont_groups(file: h5py.File) -> list[int]:
    return [cont_id_from_name(name) for name in get_cont_group_names_from_file(file)]


def get_cont_data_by_id_from_file(file: h5py.File, cont_id: int) -> np.ndarray:
    return np.array(get_cont_group_by_id_from_file(file, cont_id).get("DATA")[:])


def get_calibrated_cont_data_by_id(file: h5py.File, cont_id: int) -> np.ndarray:
    """Return calibrated data from a CONT group. If calibration attribute is
    missing, return raw data, but issue warning. The shape of the returned array
    is (nSamples, nChannels)
    """
    calibration = get_cont_group_by_id_from_file(file, cont_id).attrs.get("Calibration")
    if calibration is None:
        warnings.warn(
            DH5CalibrationMissingWarning(
                f"Calibration attribute is missing from CONT{cont_id}",
                cont_id=cont_id,
            )
        )
        return get_cont_data_by_id_from_file(file, cont_id)
    return get_cont_data_by_id_from_file(file, cont_id) * calibration


def get_cont_group_by_id_from_file(file: h5py.File, id: int) -> h5py.Group:
    contGroup = file.get(cont_name_from_id(id))
    if contGroup is None:
        raise DH5Error(f"CONT{id} does not exist in {file.filename}")
    return contGroup


def get_cont_group_names_from_file(
    filename: h5py.File,
) -> list[str]:
    cont_group_names = [
        name
        for name in filename.keys()
        if name.startswith(CONT_PREFIX) and isinstance(filename[name], h5py.Group)
    ]
    cont_group_names.sort(key=lambda name: int(name[len(CONT_PREFIX) :]))
    return cont_group_names


def get_cont_groups_from_file(file: h5py.File) -> list[h5py.Group]:
    cont_group_names = get_cont_group_names_from_file(file)
    return [file[name] for name in cont_group_names]


# validate
def validate_cont_dtype(file: h5py.File) -> None:
    # check for named data type /CONT_INDEX_ITEM
    if CONT_DTYPE_NAME not in file:
        raise DH5Error("CONT_INDEX_ITEM not found")

    # CONT_INDEX_ITEM must be a compound data type with time and offset
    cont_dtype: h5py.Datatype = file[CONT_DTYPE_NAME]
    if not isinstance(cont_dtype, h5py.Datatype) or cont_dtype.dtype.names != (
        "time",
        "offset",
    ):
        raise DH5Error(
            "CONT_INDEX_ITEM is not a named data type with fields 'time' and 'offset'"
        )


def validate_cont_group(cont_group: h5py.Group) -> None:
    """Validate a CONT group in a DAQ-HDF5 file.

    This function checks if the CONT group has the required attributes and datasets.
    """
    if not isinstance(cont_group, h5py.Group):
        raise DH5Error("Not a valid HDF5 group")

    calibration = cont_group.attrs.get("Calibration")
    if calibration is None:
        cont_id = cont_id_from_name(cont_group.name)
        warnings.warn(
            DH5CalibrationMissingWarning(
                f"Calibration attribute is missing from CONT group {cont_group.name}",
                cont_id=cont_id,
            )
        )
    else:
        if not isinstance(calibration, np.ndarray):
            raise DH5Error(
                f"Calibration attribute in {cont_group.name} is not a np array"
            )
        if not len(calibration) == cont_group["DATA"].shape[1]:
            raise DH5Error(
                f"Calibration attribute in {cont_group.name} has wrong length: {len(calibration)}. Must have length equal to number of channels"
            )

    if cont_group.attrs.get("SamplePeriod") is None:
        raise DH5Error(
            f"SamplePeriod attribute is missing from CONT group {cont_group.name}"
        )

    if DATA_DATASET_NAME not in cont_group:
        raise DH5Error(f"DATA dataset is missing from CONT group {cont_group.name}")

    data = cont_group[DATA_DATASET_NAME]
    if not isinstance(data, h5py.Dataset):
        raise DH5Error(f"DATA dataset in {cont_group.name} is not a dataset")

    # size of DATA must be (nSamples, nChannels)
    if len(data.shape) != 2:
        raise DH5Error(
            f"DATA dataset in {cont_group.name} has wrong shape: {data.shape}. Must be 2D"
        )

    if data.dtype != np.int16:
        raise DH5Error(
            f"DATA dataset in {cont_group.name} has wrong dtype: {data.dtype}. Must be int16"
        )

    if INDEX_DATASET_NAME not in cont_group:
        raise DH5Error(f"INDEX dataset is missing from CONT group {cont_group.name}")

    # INDEX must be a compound dataset with fields 'time' and 'offset'
    if not isinstance(cont_group[INDEX_DATASET_NAME], h5py.Dataset) or cont_group[
        INDEX_DATASET_NAME
    ].dtype.names != (
        "time",
        "offset",
    ):
        raise DH5Error(
            f"INDEX dataset in {cont_group.name} is not a named data type with fields 'time' and 'offset'"
        )

    if "Channels" in cont_group.attrs:
        channels = cont_group.attrs.get("Channels")
        if not isinstance(channels, np.ndarray):
            raise DH5Error(f"Channels attribute in {cont_group.name} is not a np array")
        if not channels.dtype.names == (
            "GlobalChanNumber",
            "BoardChanNo",
            "ADCBitWidth",
            "MaxVoltageRange",
            "MinVoltageRange",
            "AmplifChan0",
        ):
            raise DH5Error(
                f"Channels attribute in {cont_group.name} has wrong dtype: {channels.dtype}. Must have fields 'GlobalChanNumber', 'BoardChanNo', 'ADCBitWidth', 'MaxVoltageRange', 'MinVoltageRange', 'AmplifyChan0'"
            )
    else:
        # should be an error according to specification, but is often missing
        cont_id = cont_id_from_name(cont_group.name)
        warnings.warn(
            DH5ChannelsMissingWarning(
                f"Channels attribute is missing from CONT group {cont_group.name}",
                cont_id=cont_id,
            )
        )
