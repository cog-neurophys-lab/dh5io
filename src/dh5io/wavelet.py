"""Wavelet data in WAVELET blocks of DAQ-HDF files.

WAVELET blocks store time-frequency analysis data, typically obtained through
wavelet transforms of continuous signals. Each block represents a multi-channel
nTrode with time-frequency decomposition data.

WAVELET blocks are stored in groups named `WAVELETn`, where n can have values
between 0 and 65535. `CONT`, `SPIKE`, and `WAVELET` blocks can share the same
identifier numbers.

`WAVELETn` group **must** have the following **attributes**:

- `SamplePeriod` (`int32` scalar) - specified in nanoseconds. It's the time
  interval between two consecutive time samples in the wavelet analysis.

- `FrequencyAxis` (`double` array[F]) - specifies the frequency values (in Hz)
  corresponding to each frequency bin in the wavelet analysis.

`WAVELETn` group **must** have the following **datasets**:

- `DATA` (`struct` array[N,M,F]) with compound dtype containing:
  - `a` (amplitude/magnitude) as uint16 (0-65535)
  - `phi` (phase) as int8 (-127 to 127)

- `INDEX` (`struct` array[R]) with compound dtype containing:
  - `time` (int64) - timestamp of first sample in nanoseconds
  - `offset` (int64) - sample offset within DATA dataset (1-based)
  - `scaling` (float64) - scaling factor for amplitude values

Where N is the number of channels, M is the number of time samples,
F is the number of frequency bins, and R is the number of recording regions.
"""

import logging
import warnings

import h5py
import numpy as np
import numpy.typing as npt

from dh5io.errors import DH5Error, DH5Warning
from dh5io.hdf5_strings import ascii_str, decode_str_attr
from dhspec.wavelet import (
    DATA_DATASET_NAME,
    DATA_DTYPE,
    INDEX_DATASET_NAME,
    INDEX_DTYPE,
    WAVELET_DTYPE_NAME,
    WAVELET_PREFIX,
    FrequencyAxisType,
    amplitude_to_float,
    create_empty_index_array,
    float_to_amplitude,
    phase_to_radians,
    radians_to_phase,
    wavelet_id_from_name,
    wavelet_name_from_id,
)

logger = logging.getLogger(__name__)


class Wavelet:
    """Represents a WAVELET block in a DAQ-HDF file."""

    def __init__(self, group: h5py.Group):
        self._group = group

    def __str__(self) -> str:
        return (
            f"Wavelet(id={self.id}, "
            f"name={self.name}, "
            f"n_channels={self.n_channels}, "
            f"n_samples={self.n_samples}, "
            f"n_frequencies={self.n_frequencies}, "
            f"n_regions={self.n_regions}, "
            f"sample_period={self.sample_period}ns, "
            f"frequency_range={self.frequency_axis[0]:.2f}-{self.frequency_axis[-1]:.2f}Hz)"
        )

    def __repr__(self) -> str:
        return f"Wavelet({self._group.name})"

    @property
    def id(self) -> int:
        """Get the wavelet block ID."""
        return wavelet_id_from_name(self._group.name)

    @property
    def data(self) -> h5py.Dataset:
        """Get the DATA dataset."""
        return self._group[DATA_DATASET_NAME]

    @property
    def index(self) -> h5py.Dataset:
        """Get the INDEX dataset."""
        return self._group[INDEX_DATASET_NAME]

    @property
    def sample_period(self) -> np.int32:
        """Get the sample period in nanoseconds."""
        return self._group.attrs["SamplePeriod"]

    @property
    def frequency_axis(self) -> np.ndarray:
        """Get the frequency axis in Hz."""
        return self._group.attrs["FrequencyAxis"]

    @property
    def name(self) -> str:
        """Get the name attribute."""
        if "Name" in self._group.attrs:
            return decode_str_attr(self._group.attrs["Name"])
        return self._group.name

    @property
    def comment(self) -> str:
        """Get the comment attribute."""
        if "Comment" in self._group.attrs:
            return decode_str_attr(self._group.attrs["Comment"])
        return ""

    @property
    def n_channels(self) -> int:
        """Get the number of channels."""
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        """Get the total number of time samples."""
        return self.data.shape[1]

    @property
    def n_frequencies(self) -> int:
        """Get the number of frequency bins."""
        return self.data.shape[2]

    @property
    def n_regions(self) -> int:
        """Get the number of recording regions."""
        return self.index.shape[0]

    @property
    def duration_s(self) -> float:
        """Get the total duration in seconds.

        Calculates the duration from the first timestamp to the last timestamp
        plus the last sample's time.
        """
        if self.n_regions == 0:
            return 0.0

        index_data = self.index[:]
        first_time = index_data[0]["time"]

        # Find the last region
        last_region_idx = self.n_regions - 1
        last_time = index_data[last_region_idx]["time"]

        # Calculate how many samples in the last region
        if last_region_idx < self.n_regions - 1:
            # Not the last region - use next offset
            samples_in_last = (
                index_data[last_region_idx + 1]["offset"]
                - index_data[last_region_idx]["offset"]
            )
        else:
            # Last region - use total samples
            samples_in_last = self.n_samples - index_data[last_region_idx]["offset"] + 1

        # Duration of last region
        last_duration = samples_in_last * self.sample_period

        total_duration_ns = (last_time - first_time) + last_duration
        return total_duration_ns / 1e9

    def get_amplitude_calibrated(
        self,
        chan_idx: int | None = None,
        time_idx: int | None = None,
        freq_idx: int | None = None,
    ) -> np.ndarray:
        """Get calibrated amplitude data.

        Args:
            chan_idx: Channel index or slice. If None, returns all channels.
            time_idx: Time index or slice. If None, returns all time samples.
            freq_idx: Frequency index or slice. If None, returns all frequencies.

        Returns:
            Calibrated amplitude values as float array with shape matching the input slicing.
        """
        # Handle indexing
        if chan_idx is None:
            chan_idx = slice(None)
        if time_idx is None:
            time_idx = slice(None)
        if freq_idx is None:
            freq_idx = slice(None)

        # Get raw data with shape [N, M, F]
        raw_data = self.data[chan_idx, time_idx, freq_idx]

        # Get amplitude values
        amplitude = raw_data["a"]

        # Apply scaling from INDEX
        # Need to determine which region each sample belongs to
        index_data = self.index[:]

        # For simplicity, if getting all data or large slices,
        # process region by region
        if isinstance(time_idx, slice) and time_idx == slice(None):
            # Process all data
            calibrated = np.zeros(amplitude.shape, dtype=np.float64)
            for i in range(self.n_regions):
                offset = index_data[i]["offset"]  # Use 0-based indexing
                if i < self.n_regions - 1:
                    next_offset = index_data[i + 1]["offset"]
                else:
                    next_offset = self.n_samples

                scaling = index_data[i]["scaling"]
                # Shape is [N, M, F]
                calibrated[:, offset:next_offset, :] = amplitude_to_float(
                    amplitude[:, offset:next_offset, :], scaling
                )
            return calibrated
        else:
            # For specific indices, find the appropriate scaling
            # This is simplified - assumes single region or uses first scaling
            scaling = index_data[0]["scaling"]
            return amplitude_to_float(amplitude, scaling)

    def get_phase_radians(
        self,
        chan_idx: int | None = None,
        time_idx: int | None = None,
        freq_idx: int | None = None,
    ) -> np.ndarray:
        """Get phase data in radians.

        Args:
            chan_idx: Channel index or slice. If None, returns all channels.
            time_idx: Time index or slice. If None, returns all time samples.
            freq_idx: Frequency index or slice. If None, returns all frequencies.

        Returns:
            Phase values in radians.
        """
        if chan_idx is None:
            chan_idx = slice(None)
        if time_idx is None:
            time_idx = slice(None)
        if freq_idx is None:
            freq_idx = slice(None)

        raw_data = self.data[chan_idx, time_idx, freq_idx]
        return phase_to_radians(raw_data["phi"])

    @staticmethod
    def from_group(group: h5py.Group) -> "Wavelet":
        """Create a Wavelet instance from an HDF5 group."""
        return Wavelet(group)


def create_empty_wavelet_group_in_file(
    file: h5py.File,
    wavelet_group_id: int | None,
    n_channels: int,
    n_samples: int,
    n_frequencies: int,
    sample_period_ns: np.int32,
    frequency_axis: FrequencyAxisType,
    n_index_items: int = 1,
    name: str | None = None,
    comment: str | None = None,
) -> h5py.Group:
    """Create an empty WAVELET group in a DH5 file.

    Args:
        file: HDF5 file object
        wavelet_group_id: Wavelet block identifier (0-65535). If None, auto-assign.
        n_channels: Number of channels (N)
        n_samples: Total number of time samples (M)
        n_frequencies: Number of frequency bins (F)
        sample_period_ns: Sample period in nanoseconds
        frequency_axis: Array of frequency values in Hz (length F)
        n_index_items: Number of recording regions (default: 1)
        name: Optional name for the wavelet block
        comment: Optional comment

    Returns:
        Created HDF5 group
    """
    existing_wavelet_ids = enumerate_wavelet_groups(file)

    # Check if opened with write access
    if file.mode not in ["r+", "w", "a"]:
        raise DH5Error(
            f"File must be opened with write access but is open with {file.mode}"
        )

    # Fail if WAVELET group already exists
    if wavelet_group_id is not None and wavelet_group_id in existing_wavelet_ids:
        raise DH5Error(f"WAVELET{wavelet_group_id} already exists in {file.filename}")

    if wavelet_group_id is None:
        if len(existing_wavelet_ids) == 0:
            wavelet_group_id = 0
        else:
            wavelet_group_id = np.max(np.array(existing_wavelet_ids)) + 1
        logger.debug(
            f"No WAVELET group id provided, creating new WAVELET group {wavelet_group_id}"
        )

    # Create the WAVELET_INDEX_ITEM dtype in the file if it doesn't exist
    if WAVELET_DTYPE_NAME not in file:
        file[WAVELET_DTYPE_NAME] = INDEX_DTYPE

    wavelet_group = file.create_group(wavelet_name_from_id(wavelet_group_id))

    # Create DATA dataset with compound dtype, shape is [N, M, F]
    wavelet_group.create_dataset(
        DATA_DATASET_NAME,
        shape=(n_channels, n_samples, n_frequencies),
        dtype=DATA_DTYPE,
    )

    # Create INDEX dataset
    wavelet_group.create_dataset(
        INDEX_DATASET_NAME, shape=(n_index_items,), dtype=file[WAVELET_DTYPE_NAME]
    )

    # Set required attributes
    wavelet_group.attrs["SamplePeriod"] = np.int32(sample_period_ns)
    wavelet_group.attrs["FrequencyAxis"] = frequency_axis

    # Set optional attributes
    if name is not None:
        wavelet_group.attrs["Name"] = ascii_str(name)
    else:
        wavelet_group.attrs["Name"] = ascii_str(f"WAVELET{wavelet_group_id}")
        logger.debug(
            f"Name attribute not provided, using default {wavelet_group.attrs['Name']}"
        )

    if comment is not None:
        wavelet_group.attrs["Comment"] = ascii_str(comment)
    else:
        wavelet_group.attrs["Comment"] = ascii_str("")

    return wavelet_group


def create_wavelet_group_from_data_in_file(
    file: h5py.File,
    wavelet_group_id: int | None,
    amplitude: np.ndarray,
    phase: np.ndarray,
    index: np.ndarray,
    sample_period_ns: np.int32,
    frequency_axis: FrequencyAxisType,
    name: str | None = None,
    comment: str | None = None,
) -> h5py.Group:
    """Create a WAVELET group from data in a DH5 file.

    Args:
        file: HDF5 file object
        wavelet_group_id: Wavelet block identifier (0-65535). If None, auto-assign.
        amplitude: Amplitude data as float array, shape (N, M, F)
        phase: Phase data in radians as float array, shape (N, M, F)
        index: Index array with INDEX_DTYPE, shape (R,)
        sample_period_ns: Sample period in nanoseconds
        frequency_axis: Array of frequency values in Hz (length F)
        name: Optional name for the wavelet block
        comment: Optional comment

    Returns:
        Created HDF5 group
    """
    if amplitude.shape != phase.shape:
        raise DH5Error(
            f"Amplitude and phase shapes must match: {amplitude.shape} != {phase.shape}"
        )

    n_channels, n_samples, n_frequencies = amplitude.shape

    if len(frequency_axis) != n_frequencies:
        raise DH5Error(
            f"FrequencyAxis length {len(frequency_axis)} must match "
            f"number of frequencies {n_frequencies}"
        )

    wavelet_group = create_empty_wavelet_group_in_file(
        file,
        wavelet_group_id,
        n_channels=n_channels,
        n_samples=n_samples,
        n_frequencies=n_frequencies,
        sample_period_ns=sample_period_ns,
        frequency_axis=frequency_axis,
        n_index_items=index.shape[0],
        name=name,
        comment=comment,
    )

    # Convert data to storage format, shape is [N, M, F]
    data = np.zeros((n_channels, n_samples, n_frequencies), dtype=DATA_DTYPE)

    # Process each region with its scaling factor
    for i in range(index.shape[0]):
        offset = index[i]["offset"]  # Use 0-based indexing
        if i < index.shape[0] - 1:
            next_offset = index[i + 1]["offset"]
        else:
            next_offset = n_samples

        scaling = index[i]["scaling"]

        # Convert amplitude to uint16 using scaling, shape is [N, M, F]
        data["a"][:, offset:next_offset, :] = float_to_amplitude(
            amplitude[:, offset:next_offset, :], scaling
        )

    # Convert phase to int8 (independent of regions)
    data["phi"][:, :, :] = radians_to_phase(phase)

    # Write data
    wavelet_group[DATA_DATASET_NAME][:] = data
    wavelet_group[INDEX_DATASET_NAME][:] = index

    return wavelet_group


def enumerate_wavelet_groups(file: h5py.File) -> list[int]:
    """Enumerate all WAVELET group IDs in the file.

    Args:
        file: HDF5 file object

    Returns:
        List of wavelet block IDs
    """
    names = get_wavelet_group_names_from_file(file)
    return [wavelet_id_from_name(name) for name in names]


def get_wavelet_group_names_from_file(file: h5py.File) -> list[str]:
    """Get all WAVELET group names from the file.

    Args:
        file: HDF5 file object

    Returns:
        List of wavelet group names
    """
    return [
        name
        for name in file.keys()
        if name.startswith(WAVELET_PREFIX) and isinstance(file[name], h5py.Group)
    ]


def get_wavelet_groups_from_file(file: h5py.File) -> list[h5py.Group]:
    """Get all WAVELET groups from the file.

    Args:
        file: HDF5 file object

    Returns:
        List of HDF5 Group objects
    """
    return [file[name] for name in get_wavelet_group_names_from_file(file)]


def get_wavelet_group_by_id_from_file(file: h5py.File, id: int) -> h5py.Group | None:
    """Get a WAVELET group by ID from the file.

    Args:
        file: HDF5 file object
        id: Wavelet block identifier

    Returns:
        HDF5 Group object or None if not found
    """
    name = wavelet_name_from_id(id)
    if name in file:
        return file[name]
    return None


def validate_wavelet_dtype(file: h5py.File) -> bool:
    """Validate that the WAVELET_INDEX_ITEM dtype is correctly defined.

    Args:
        file: HDF5 file object

    Returns:
        True if valid

    Raises:
        DH5Error: If dtype is invalid
    """
    if WAVELET_DTYPE_NAME not in file:
        raise DH5Error(f"{WAVELET_DTYPE_NAME} dtype not found in file")

    dtype = file[WAVELET_DTYPE_NAME]
    expected_dtype = INDEX_DTYPE

    if dtype.dtype != expected_dtype:
        raise DH5Error(
            f"{WAVELET_DTYPE_NAME} dtype mismatch: expected {expected_dtype}, "
            f"got {dtype.dtype}"
        )

    return True


def validate_wavelet_group(group: h5py.Group) -> bool:
    """Validate a WAVELET group structure.

    Args:
        group: HDF5 Group object

    Returns:
        True if valid

    Raises:
        DH5Error: If group structure is invalid
    """
    # Check required attributes
    required_attrs = ["SamplePeriod", "FrequencyAxis"]
    for attr in required_attrs:
        if attr not in group.attrs:
            raise DH5Error(f"WAVELET group missing required attribute: {attr}")

    # Check attribute types
    if not isinstance(group.attrs["SamplePeriod"], (int, np.integer)):
        raise DH5Error("SamplePeriod must be an integer")

    frequency_axis = group.attrs["FrequencyAxis"]
    if not isinstance(frequency_axis, np.ndarray):
        raise DH5Error("FrequencyAxis must be a numpy array")

    # Check required datasets
    required_datasets = [DATA_DATASET_NAME, INDEX_DATASET_NAME]
    for dataset in required_datasets:
        if dataset not in group:
            raise DH5Error(f"WAVELET group missing required dataset: {dataset}")

    # Check DATA dataset
    data = group[DATA_DATASET_NAME]
    if len(data.shape) != 3:
        raise DH5Error(f"DATA dataset must be 3-dimensional, got shape {data.shape}")

    if data.dtype != DATA_DTYPE:
        raise DH5Error(
            f"DATA dataset dtype mismatch: expected {DATA_DTYPE}, got {data.dtype}"
        )

    n_frequencies = data.shape[2]  # Third dimension in [N, M, F]
    if len(frequency_axis) != n_frequencies:
        raise DH5Error(
            f"FrequencyAxis length {len(frequency_axis)} must match "
            f"DATA third dimension {n_frequencies}"
        )

    # Check INDEX dataset
    index = group[INDEX_DATASET_NAME]
    if len(index.shape) != 1:
        raise DH5Error(f"INDEX dataset must be 1-dimensional, got shape {index.shape}")

    if index.dtype != INDEX_DTYPE:
        raise DH5Error(
            f"INDEX dataset dtype mismatch: expected {INDEX_DTYPE}, got {index.dtype}"
        )

    return True
