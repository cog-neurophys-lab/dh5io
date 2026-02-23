import h5py
import numpy as np
import pytest

import dh5io
import dh5io.wavelet as wavelet
from dh5io.create import create_dh_file
from dhspec.wavelet import (
    amplitude_to_float,
    create_empty_index_array,
    float_to_amplitude,
    phase_to_radians,
    radians_to_phase,
)


def test_phase_conversion():
    """Test phase conversion between radians and int8."""
    # Test conversion roundtrip
    phi_rad = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2, -np.pi])
    phi_int8 = radians_to_phase(phi_rad)
    phi_rad_back = phase_to_radians(phi_int8)

    # Should be close (small rounding error expected)
    np.testing.assert_allclose(phi_rad, phi_rad_back, atol=0.03)

    # Test that int8 values are in valid range
    assert np.all(phi_int8 >= -127)
    assert np.all(phi_int8 <= 127)


def test_amplitude_conversion():
    """Test amplitude conversion between float and uint16."""
    # Test with known scaling factor
    scaling = 0.001
    a_float = np.array([0.0, 0.5, 1.0, 10.0, 65.0])
    a_uint16 = float_to_amplitude(a_float, scaling)
    a_float_back = amplitude_to_float(a_uint16, scaling)

    # Should be close
    np.testing.assert_allclose(a_float, a_float_back, rtol=0.01)

    # Test that uint16 values are in valid range
    assert np.all(a_uint16 >= 0)
    assert np.all(a_uint16 <= 65535)


def test_create_empty_wavelet_group(tmp_path):
    """Test creating an empty wavelet group."""
    filename = tmp_path / "test.dh5"
    wavelet_group_name = "test_wavelet"
    wavelet_group_id = 100
    sample_period_ns = 1_000_000  # 1 ms
    n_channels = 4
    n_samples = 10000
    n_frequencies = 50
    n_index_items = 3
    frequency_axis = np.logspace(1, 2, n_frequencies)  # 10-100 Hz

    with create_dh_file(filename) as dh5file:
        wavelet_group = wavelet.create_empty_wavelet_group_in_file(
            dh5file._file,
            wavelet_group_id=wavelet_group_id,
            n_channels=n_channels,
            n_samples=n_samples,
            n_frequencies=n_frequencies,
            sample_period_ns=sample_period_ns,
            frequency_axis=frequency_axis,
            n_index_items=n_index_items,
            name=wavelet_group_name,
        )

        assert wavelet_group.attrs["SamplePeriod"] == sample_period_ns
        np.testing.assert_array_equal(
            wavelet_group.attrs["FrequencyAxis"], frequency_axis
        )
        assert wavelet_group.attrs["Name"] == wavelet_group_name.encode()

    # Verify by reading back
    with dh5io.DH5File(filename, "r") as dh5file:
        wavelet_group = wavelet.get_wavelet_group_by_id_from_file(
            dh5file._file, wavelet_group_id
        )

        assert wavelet_group is not None
        assert wavelet_group.attrs["SamplePeriod"] == sample_period_ns
        np.testing.assert_array_equal(
            wavelet_group.attrs["FrequencyAxis"], frequency_axis
        )
        assert wavelet_group["DATA"].shape == (n_channels, n_samples, n_frequencies)
        assert wavelet_group["INDEX"].shape == (n_index_items,)

        # Validate the group
        assert wavelet.validate_wavelet_group(wavelet_group)


def test_create_wavelet_group_with_data(tmp_path):
    """Test creating a wavelet group with actual data."""
    filename = tmp_path / "test.dh5"
    wavelet_group_name = "test_wavelet"
    wavelet_group_id = 50
    sample_period_ns = 500_000  # 0.5 ms
    n_channels = 2
    n_samples = 5000
    n_frequencies = 30
    n_index_items = 2
    frequency_axis = np.logspace(1, 2, n_frequencies)  # 10-100 Hz

    # Create synthetic wavelet data, shape [N, M, F]
    # Amplitude: random positive values
    amplitude = np.random.rand(n_channels, n_samples, n_frequencies) * 10.0

    # Phase: random values in [-pi, pi]
    phase = (np.random.rand(n_channels, n_samples, n_frequencies) - 0.5) * 2 * np.pi

    # Create index with two regions
    index = create_empty_index_array(n_index_items)
    index[0] = (0, 0, 0.0002)  # First region starts at sample 0, scaling 0.0002
    index[1] = (
        2_500_000_000,
        2500,
        0.0003,
    )  # Second region starts at sample 2500, scaling 0.0003

    with create_dh_file(filename) as dh5file:
        wavelet_group = wavelet.create_wavelet_group_from_data_in_file(
            dh5file._file,
            wavelet_group_id=wavelet_group_id,
            amplitude=amplitude,
            phase=phase,
            index=index,
            sample_period_ns=sample_period_ns,
            frequency_axis=frequency_axis,
            name=wavelet_group_name,
        )

        assert wavelet_group.attrs["SamplePeriod"] == sample_period_ns
        np.testing.assert_array_equal(
            wavelet_group.attrs["FrequencyAxis"], frequency_axis
        )
        assert wavelet_group["DATA"].shape == (n_channels, n_samples, n_frequencies)
        assert wavelet_group["INDEX"].shape == (n_index_items,)

        # Verify index data
        np.testing.assert_array_equal(wavelet_group["INDEX"][:], index)

    # Verify by reading back
    with dh5io.DH5File(filename, "r") as dh5file:
        wavelet_group = wavelet.get_wavelet_group_by_id_from_file(
            dh5file._file, wavelet_group_id
        )

        assert wavelet_group is not None
        wavelet.validate_wavelet_group(wavelet_group)

        # Read back data and verify it's close to original
        # (won't be exact due to quantization)
        data = wavelet_group["DATA"][:]

        # Check phase conversion
        phase_back = phase_to_radians(data["phi"])
        # Allow for quantization error
        assert np.allclose(phase, phase_back, atol=0.03)

        # Check amplitude for first region (samples 0-2500), shape is [N, M, F]
        scaling1 = index[0]["scaling"]
        amplitude_back_region1 = amplitude_to_float(data["a"][:, 0:2500, :], scaling1)
        assert np.allclose(
            amplitude[:, 0:2500, :], amplitude_back_region1, rtol=0.001, atol=0.001
        )

        # Check amplitude for second region (samples 2500-end)
        scaling2 = index[1]["scaling"]
        amplitude_back_region2 = amplitude_to_float(data["a"][:, 2500:, :], scaling2)
        assert np.allclose(
            amplitude[:, 2500:, :], amplitude_back_region2, rtol=0.001, atol=0.001
        )


def test_wavelet_class(tmp_path):
    """Test the Wavelet class wrapper."""
    filename = tmp_path / "test.dh5"
    wavelet_group_name = "test_wavelet"
    wavelet_group_id = 25
    sample_period_ns = 1_000_000
    n_channels = 3
    n_samples = 1000
    n_frequencies = 20
    n_index_items = 1
    frequency_axis = np.linspace(10, 100, n_frequencies)

    # Create synthetic data, shape [N, M, F]
    amplitude = np.random.rand(n_channels, n_samples, n_frequencies) * 5.0
    phase = (np.random.rand(n_channels, n_samples, n_frequencies) - 0.5) * 2 * np.pi

    index = create_empty_index_array(n_index_items)
    index[0] = (0, 0, 0.0001)

    with create_dh_file(filename) as dh5file:
        wavelet.create_wavelet_group_from_data_in_file(
            dh5file._file,
            wavelet_group_id=wavelet_group_id,
            amplitude=amplitude,
            phase=phase,
            index=index,
            sample_period_ns=sample_period_ns,
            frequency_axis=frequency_axis,
            name=wavelet_group_name,
        )

    # Test Wavelet class
    with h5py.File(filename, "r") as f:
        group = f[f"WAVELET{wavelet_group_id}"]
        wv = wavelet.Wavelet(group)

        # Test properties
        assert wv.id == wavelet_group_id
        assert wv.name == wavelet_group_name
        assert wv.sample_period == sample_period_ns
        assert wv.n_channels == n_channels
        assert wv.n_samples == n_samples
        assert wv.n_frequencies == n_frequencies
        assert wv.n_regions == n_index_items
        np.testing.assert_array_equal(wv.frequency_axis, frequency_axis)

        # Test duration calculation
        expected_duration_s = n_samples * sample_period_ns / 1e9
        assert abs(wv.duration_s - expected_duration_s) < 0.001

        # Test __str__ and __repr__
        str_repr = str(wv)
        assert "Wavelet" in str_repr
        assert str(wavelet_group_id) in str_repr

        repr_str = repr(wv)
        assert "Wavelet" in repr_str

        # Test calibrated amplitude retrieval
        amplitude_calibrated = wv.get_amplitude_calibrated()
        assert amplitude_calibrated.shape == (n_channels, n_samples, n_frequencies)
        # Should be close to original (within quantization error)
        assert np.allclose(amplitude, amplitude_calibrated, rtol=0.001, atol=0.001)

        # Test phase retrieval
        phase_radians = wv.get_phase_radians()
        assert phase_radians.shape == (n_channels, n_samples, n_frequencies)
        assert np.allclose(phase, phase_radians, atol=0.03)


def test_enumerate_wavelet_groups(tmp_path):
    """Test enumerating wavelet groups."""
    filename = tmp_path / "test.dh5"
    frequency_axis = np.linspace(10, 100, 10)

    with create_dh_file(filename) as dh5file:
        # Create multiple wavelet groups
        for wv_id in [0, 5, 10, 100]:
            wavelet.create_empty_wavelet_group_in_file(
                dh5file._file,
                wavelet_group_id=wv_id,
                n_channels=2,
                n_samples=100,
                n_frequencies=10,
                sample_period_ns=1_000_000,
                frequency_axis=frequency_axis,
            )

    with h5py.File(filename, "r") as f:
        wavelet_ids = wavelet.enumerate_wavelet_groups(f)
        assert sorted(wavelet_ids) == [0, 5, 10, 100]

        wavelet_names = wavelet.get_wavelet_group_names_from_file(f)
        assert len(wavelet_names) == 4
        assert "WAVELET0" in wavelet_names
        assert "WAVELET5" in wavelet_names
        assert "WAVELET10" in wavelet_names
        assert "WAVELET100" in wavelet_names

        wavelet_groups = wavelet.get_wavelet_groups_from_file(f)
        assert len(wavelet_groups) == 4


def test_auto_assign_wavelet_id(tmp_path):
    """Test automatic wavelet ID assignment."""
    filename = tmp_path / "test.dh5"
    frequency_axis = np.linspace(10, 100, 10)

    with create_dh_file(filename) as dh5file:
        # Create first group with auto ID (should be 0)
        wv_group1 = wavelet.create_empty_wavelet_group_in_file(
            dh5file._file,
            wavelet_group_id=None,
            n_channels=2,
            n_samples=100,
            n_frequencies=10,
            sample_period_ns=1_000_000,
            frequency_axis=frequency_axis,
        )
        assert wv_group1.name == "/WAVELET0"

        # Create second group with auto ID (should be 1)
        wv_group2 = wavelet.create_empty_wavelet_group_in_file(
            dh5file._file,
            wavelet_group_id=None,
            n_channels=2,
            n_samples=100,
            n_frequencies=10,
            sample_period_ns=1_000_000,
            frequency_axis=frequency_axis,
        )
        assert wv_group2.name == "/WAVELET1"


def test_wavelet_error_handling(tmp_path):
    """Test error handling in wavelet functions."""
    filename = tmp_path / "test.dh5"
    frequency_axis = np.linspace(10, 100, 10)

    with create_dh_file(filename) as dh5file:
        # Create a wavelet group
        wavelet.create_empty_wavelet_group_in_file(
            dh5file._file,
            wavelet_group_id=0,
            n_channels=2,
            n_samples=100,
            n_frequencies=10,
            sample_period_ns=1_000_000,
            frequency_axis=frequency_axis,
        )

        # Try to create duplicate - should raise error
        with pytest.raises(Exception):  # DH5Error
            wavelet.create_empty_wavelet_group_in_file(
                dh5file._file,
                wavelet_group_id=0,
                n_channels=2,
                n_samples=100,
                n_frequencies=10,
                sample_period_ns=1_000_000,
                frequency_axis=frequency_axis,
            )


def test_wavelet_with_mismatched_shapes(tmp_path):
    """Test error handling for mismatched amplitude/phase shapes."""
    filename = tmp_path / "test.dh5"
    frequency_axis = np.linspace(10, 100, 10)

    amplitude = np.random.rand(2, 100, 10)  # [N=2, M=100, F=10]
    phase = np.random.rand(3, 100, 10)  # Different channel count [N=3, M=100, F=10]
    index = create_empty_index_array(1)
    index[0] = (0, 0, 0.001)

    with create_dh_file(filename) as dh5file:
        with pytest.raises(Exception):  # DH5Error
            wavelet.create_wavelet_group_from_data_in_file(
                dh5file._file,
                wavelet_group_id=0,
                amplitude=amplitude,
                phase=phase,
                index=index,
                sample_period_ns=1_000_000,
                frequency_axis=frequency_axis,
            )


def test_wavelet_with_wrong_frequency_axis_length(tmp_path):
    """Test error handling for wrong frequency axis length."""
    filename = tmp_path / "test.dh5"
    frequency_axis = np.linspace(10, 100, 5)  # Wrong length

    amplitude = np.random.rand(2, 100, 10)  # [N=2, M=100, F=10]
    phase = np.random.rand(2, 100, 10)
    index = create_empty_index_array(1)
    index[0] = (0, 0, 0.001)

    with create_dh_file(filename) as dh5file:
        with pytest.raises(Exception):  # DH5Error
            wavelet.create_wavelet_group_from_data_in_file(
                dh5file._file,
                wavelet_group_id=0,
                amplitude=amplitude,
                phase=phase,
                index=index,
                sample_period_ns=1_000_000,
                frequency_axis=frequency_axis,
            )


def test_wavelet_multiregion_calibration(tmp_path):
    """Test amplitude calibration with multiple regions."""
    filename = tmp_path / "test.dh5"
    wavelet_group_id = 0
    n_channels = 2
    n_samples = 1000
    n_frequencies = 10
    frequency_axis = np.linspace(10, 100, n_frequencies)

    # Create data with different scaling in different regions, shape [N, M, F]
    amplitude = np.ones((n_channels, n_samples, n_frequencies))
    amplitude[:, :500, :] = 1.0  # First region
    amplitude[:, 500:, :] = 2.0  # Second region

    phase = np.zeros((n_channels, n_samples, n_frequencies))

    # Create index with two regions
    index = create_empty_index_array(2)
    index[0] = (0, 0, 0.001)  # Region 1: scaling 0.001
    index[1] = (500_000_000, 500, 0.002)  # Region 2: scaling 0.002

    with create_dh_file(filename) as dh5file:
        wavelet.create_wavelet_group_from_data_in_file(
            dh5file._file,
            wavelet_group_id=wavelet_group_id,
            amplitude=amplitude,
            phase=phase,
            index=index,
            sample_period_ns=1_000_000,
            frequency_axis=frequency_axis,
        )

    # Read back and verify
    with h5py.File(filename, "r") as f:
        wv = wavelet.Wavelet(f[f"WAVELET{wavelet_group_id}"])

        # Get calibrated amplitude
        amplitude_calibrated = wv.get_amplitude_calibrated()

        # Verify both regions are correctly scaled
        assert np.allclose(
            amplitude_calibrated[:, :500, :], 1.0, rtol=0.001, atol=0.001
        )
        assert np.allclose(
            amplitude_calibrated[:, 500:, :], 2.0, rtol=0.001, atol=0.001
        )


def test_wavelet_integration_with_dh5file(tmp_path):
    """Test wavelet integration with DH5File class."""
    filename = tmp_path / "test.dh5"
    wavelet_group_name = "integrated_wavelet"
    wavelet_group_id = 10
    sample_period_ns = 1_000_000
    n_channels = 2
    n_samples = 500
    n_frequencies = 15
    frequency_axis = np.logspace(1, 2, n_frequencies)

    # Create synthetic data, shape [N, M, F]
    amplitude = np.random.rand(n_channels, n_samples, n_frequencies) * 3.0
    phase = (np.random.rand(n_channels, n_samples, n_frequencies) - 0.5) * 2 * np.pi

    index = create_empty_index_array(1)
    index[0] = (0, 0, 0.0001)

    # Create file with wavelet data
    with create_dh_file(filename) as dh5file:
        wavelet.create_wavelet_group_from_data_in_file(
            dh5file._file,
            wavelet_group_id=wavelet_group_id,
            amplitude=amplitude,
            phase=phase,
            index=index,
            sample_period_ns=sample_period_ns,
            frequency_axis=frequency_axis,
            name=wavelet_group_name,
        )

    # Test reading through DH5File
    with dh5io.DH5File(filename, "r") as dh5file:
        # Test enumeration methods
        wavelet_ids = dh5file.get_wavelet_group_ids()
        assert wavelet_group_id in wavelet_ids

        wavelet_names = dh5file.get_wavelet_group_names()
        assert f"WAVELET{wavelet_group_id}" in wavelet_names

        wavelet_groups = dh5file.get_wavelet_groups()
        assert len(wavelet_groups) == 1

        # Test getting by ID
        wv = dh5file.get_wavelet_group_by_id(wavelet_group_id)
        assert wv is not None
        assert isinstance(wv, wavelet.Wavelet)
        assert wv.id == wavelet_group_id
        assert wv.name == wavelet_group_name
        assert wv.n_channels == n_channels
        assert wv.n_samples == n_samples
        assert wv.n_frequencies == n_frequencies

        # Verify data can be retrieved
        amplitude_calibrated = wv.get_amplitude_calibrated()
        assert amplitude_calibrated.shape == (n_channels, n_samples, n_frequencies)

        phase_radians = wv.get_phase_radians()
        assert phase_radians.shape == (n_channels, n_samples, n_frequencies)

        # Test __str__ includes wavelet info
        dh5_str = str(dh5file)
        assert "WAVELET" in dh5_str
        assert (
            str(wavelet_group_id) in dh5_str or f"WAVELET{wavelet_group_id}" in dh5_str
        )
