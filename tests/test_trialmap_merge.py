"""Test merging TRIALMAPs from multiple DH5 files."""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dh5cli.dh5merge import merge_dh5_files
from dh5io.cont import create_cont_group_from_data_in_file, create_empty_index_array
from dh5io.create import create_dh_file
from dh5io.dh5file import DH5File
from dh5io.trialmap import add_trialmap_to_file
from dhspec.trialmap import TRIALMAP_DATASET_DTYPE


def create_test_trialmap(
    n_trials: int, start_trial_no: int = 0, start_time: int = 0
) -> np.recarray:
    """Create a test TRIALMAP with n_trials."""
    trialmap = np.recarray(n_trials, dtype=TRIALMAP_DATASET_DTYPE)

    time_per_trial = 1_000_000_000  # 1 second per trial in nanoseconds

    for i in range(n_trials):
        trialmap[i].TrialNo = start_trial_no + i
        trialmap[i].StimNo = (i % 5) + 1  # Cycle through stim numbers 1-5
        trialmap[i].Outcome = 1 if i % 2 == 0 else 0  # Alternate outcomes
        trialmap[i].StartTime = start_time + i * time_per_trial
        trialmap[i].EndTime = start_time + (i + 1) * time_per_trial

    return trialmap


def test_merge_trialmaps():
    """Test merging TRIALMAPs from multiple files."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create three test files with TRIALMAPs
        file1 = tmpdir / "test1.dh5"
        file2 = tmpdir / "test2.dh5"
        file3 = tmpdir / "test3.dh5"
        output = tmpdir / "merged.dh5"

        n_trials = [10, 15, 8]

        # Create file 1 with 10 trials
        with create_dh_file(file1, overwrite=True) as dh5:
            # Add a CONT block
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            # Add TRIALMAP
            trialmap = create_test_trialmap(n_trials[0], start_trial_no=1, start_time=0)
            add_trialmap_to_file(dh5._file, trialmap)

        # Create file 2 with 15 trials
        with create_dh_file(file2, overwrite=True) as dh5:
            # Add a CONT block
            data = np.random.randint(-1000, 1000, size=(150, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (10_000_000_000, 0)  # 10 seconds later
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            # Add TRIALMAP
            trialmap = create_test_trialmap(
                n_trials[1], start_trial_no=11, start_time=10_000_000_000
            )
            add_trialmap_to_file(dh5._file, trialmap)

        # Create file 3 with 8 trials
        with create_dh_file(file3, overwrite=True) as dh5:
            # Add a CONT block
            data = np.random.randint(-1000, 1000, size=(80, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (25_000_000_000, 0)  # 25 seconds later
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            # Add TRIALMAP
            trialmap = create_test_trialmap(
                n_trials[2], start_trial_no=26, start_time=25_000_000_000
            )
            add_trialmap_to_file(dh5._file, trialmap)

        # Merge files
        merge_dh5_files([file1, file2, file3], output, overwrite=True)

        # Verify the merged TRIALMAP
        with DH5File(output, mode="r") as merged:
            merged_trialmap = merged.get_trialmap()

            assert merged_trialmap is not None, "Merged file should have TRIALMAP"

            total_trials = sum(n_trials)
            assert len(merged_trialmap) == total_trials, (
                f"Expected {total_trials} trials, got {len(merged_trialmap)}"
            )

            # Check first trial from file 1
            assert merged_trialmap.trial_numbers[0] == 1
            assert merged_trialmap.trial_type_numbers[0] == 1

            # Check first trial from file 2
            assert merged_trialmap.trial_numbers[n_trials[0]] == 11
            assert merged_trialmap.start_time_nanoseconds[n_trials[0]] == 10_000_000_000

            # Check first trial from file 3
            file3_start_idx = n_trials[0] + n_trials[1]
            assert merged_trialmap[file3_start_idx].TrialNo == 26
            assert merged_trialmap[file3_start_idx].StartTime == 25_000_000_000

            # Check last trial
            assert merged_trialmap[-1].TrialNo == 26 + n_trials[2] - 1


def test_merge_with_missing_trialmaps():
    """Test merging files where some have TRIALMAPs and some don't."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        file1 = tmpdir / "test1.dh5"
        file2 = tmpdir / "test2.dh5"
        file3 = tmpdir / "test3.dh5"
        output = tmpdir / "merged.dh5"

        # Create file 1 with TRIALMAP
        with create_dh_file(file1, overwrite=True) as dh5:
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            trialmap = create_test_trialmap(5, start_trial_no=1, start_time=0)
            add_trialmap_to_file(dh5._file, trialmap)

        # Create file 2 WITHOUT TRIALMAP
        with create_dh_file(file2, overwrite=True) as dh5:
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (5_000_000_000, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )
            # No TRIALMAP

        # Create file 3 with TRIALMAP
        with create_dh_file(file3, overwrite=True) as dh5:
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (10_000_000_000, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            trialmap = create_test_trialmap(
                7, start_trial_no=6, start_time=10_000_000_000
            )
            add_trialmap_to_file(dh5._file, trialmap)

        # Merge files
        merge_dh5_files([file1, file2, file3], output, overwrite=True)

        # Verify the merged TRIALMAP (should have 5 + 7 = 12 trials)
        with DH5File(output, mode="r") as merged:
            merged_trialmap = merged.get_trialmap()

            assert merged_trialmap is not None, "Merged file should have TRIALMAP"
            assert len(merged_trialmap) == 12, (
                f"Expected 12 trials, got {len(merged_trialmap)}"
            )

            # Check first trial from file 1
            assert merged_trialmap[0].TrialNo == 1

            # Check first trial from file 3 (file 2 had no trialmap)
            assert merged_trialmap[5].TrialNo == 6


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Merging TRIALMAPs from all files")
    print("=" * 60)
    success1 = test_merge_trialmaps()

    print()
    print("=" * 60)
    print("Test 2: Merging with some files missing TRIALMAPs")
    print("=" * 60)
    success2 = test_merge_with_missing_trialmaps()

    print()
    if success1 and success2:
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("✗ Some tests failed")
        print("=" * 60)
        sys.exit(1)
