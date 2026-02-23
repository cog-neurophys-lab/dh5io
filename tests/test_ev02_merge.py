"""Test merging EV02 (event triggers) from multiple DH5 files."""

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
from dh5io.event_triggers import (
    add_event_triggers_to_file,
    get_event_triggers_from_file,
)
from dh5io.operations import get_operations_group


def create_test_events(
    n_events: int, start_time: int = 0, start_event_code: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Create test event triggers."""
    timestamps = np.arange(n_events, dtype=np.int64) * 1_000_000 + start_time
    event_codes = np.arange(n_events, dtype=np.int32) % 10 + start_event_code
    return timestamps, event_codes


def test_merge_event_triggers():
    """Test merging EV02 datasets from multiple files."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create three test files with EV02
        file1 = tmpdir / "test1.dh5"
        file2 = tmpdir / "test2.dh5"
        file3 = tmpdir / "test3.dh5"
        output = tmpdir / "merged.dh5"

        n_events = [20, 30, 15]

        # Create file 1 with 20 events
        with create_dh_file(file1, overwrite=True) as dh5:
            # Add a CONT block
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            # Add EV02
            timestamps, event_codes = create_test_events(
                n_events[0], start_time=0, start_event_code=1
            )
            add_event_triggers_to_file(dh5._file, timestamps, event_codes)

        # Create file 2 with 30 events
        with create_dh_file(file2, overwrite=True) as dh5:
            # Add a CONT block
            data = np.random.randint(-1000, 1000, size=(150, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (20_000_000, 0)  # 20ms later
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            # Add EV02
            timestamps, event_codes = create_test_events(
                n_events[1], start_time=20_000_000, start_event_code=5
            )
            add_event_triggers_to_file(dh5._file, timestamps, event_codes)

        # Create file 3 with 15 events
        with create_dh_file(file3, overwrite=True) as dh5:
            # Add a CONT block
            data = np.random.randint(-1000, 1000, size=(80, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (50_000_000, 0)  # 50ms later
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            # Add EV02
            timestamps, event_codes = create_test_events(
                n_events[2], start_time=50_000_000, start_event_code=3
            )
            add_event_triggers_to_file(dh5._file, timestamps, event_codes)

        # Merge files
        merge_dh5_files([file1, file2, file3], output, overwrite=True)

        # Verify the merged EV02
        with DH5File(output, mode="r") as merged:
            merged_events = get_event_triggers_from_file(merged._file)

            assert merged_events is not None, "Merged file should have EV02"

            total_events = sum(n_events)
            assert len(merged_events) == total_events, (
                f"Expected {total_events} events, got {len(merged_events)}"
            )

            # Check first event from file 1
            assert merged_events[0]["time"] == 0
            assert merged_events[0]["event"] == 1

            # Check first event from file 2
            assert merged_events[n_events[0]]["time"] == 20_000_000
            assert merged_events[n_events[0]]["event"] == 5

            # Check first event from file 3
            file3_start_idx = n_events[0] + n_events[1]
            assert merged_events[file3_start_idx]["time"] == 50_000_000
            assert merged_events[file3_start_idx]["event"] == 3


def test_merge_operation_recorded():
    """Test that merge operation is recorded in the output file."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        file1 = tmpdir / "test1.dh5"
        file2 = tmpdir / "test2.dh5"
        output = tmpdir / "merged.dh5"

        # Create two simple files with sequential timestamps
        # File 1: starts at 0
        with create_dh_file(file1, overwrite=True) as dh5:
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

        # File 2: starts after file 1 ends (100 samples * 1000 ns = 100,000 ns)
        with create_dh_file(file2, overwrite=True) as dh5:
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (200_000, 0)  # Start after file 1
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

        # Merge files
        merge_dh5_files([file1, file2], output, overwrite=True)

        # Verify operation was recorded
        with DH5File(output, mode="r") as merged:
            operations_group = get_operations_group(merged._file)

            assert operations_group is not None, (
                "Output file should have Operations group"
            )

            # Check that there's at least a merge operation
            operation_names = list(operations_group.keys())
            merge_operations = [
                name for name in operation_names if "merge" in name.lower()
            ]

            assert len(merge_operations) > 0, "Should have at least one merge operation"

            # Get the merge operation
            merge_op = operations_group[merge_operations[0]]

            # Check attributes
            assert "Tool" in merge_op.attrs, (
                "Merge operation should have Tool attribute"
            )
            tool_val = merge_op.attrs["Tool"]
            if isinstance(tool_val, bytes):
                tool_val = tool_val.decode()
            assert "dh5merge" in tool_val, "Tool should mention dh5merge"

            if "MergedFiles" in merge_op.attrs:
                merged_files = merge_op.attrs["MergedFiles"]
                print(f"  Merged files: {list(merged_files)}")

            if "NumberOfFiles" in merge_op.attrs:
                num_files = merge_op.attrs["NumberOfFiles"]
                assert num_files == 2, f"Should have merged 2 files, got {num_files}"
                print(f"  Number of files: {num_files}")


def test_merge_with_missing_ev02():
    """Test merging when some files don't have EV02."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        file1 = tmpdir / "test1.dh5"
        file2 = tmpdir / "test2.dh5"
        file3 = tmpdir / "test3.dh5"
        output = tmpdir / "merged.dh5"

        # File 1 with EV02
        with create_dh_file(file1, overwrite=True) as dh5:
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            timestamps, event_codes = create_test_events(10, start_time=0)
            add_event_triggers_to_file(dh5._file, timestamps, event_codes)

        # File 2 WITHOUT EV02
        with create_dh_file(file2, overwrite=True) as dh5:
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (10_000_000, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )
            # No EV02

        # File 3 with EV02
        with create_dh_file(file3, overwrite=True) as dh5:
            data = np.random.randint(-1000, 1000, size=(100, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (20_000_000, 0)
            create_cont_group_from_data_in_file(
                dh5._file, 0, data, index, np.int32(1000)
            )

            timestamps, event_codes = create_test_events(8, start_time=20_000_000)
            add_event_triggers_to_file(dh5._file, timestamps, event_codes)

        # Merge files
        merge_dh5_files([file1, file2, file3], output, overwrite=True)

        # Verify the merged EV02 (should have 10 + 8 = 18 events)
        with DH5File(output, mode="r") as merged:
            merged_events = get_event_triggers_from_file(merged._file)

            assert merged_events is not None, "Merged file should have EV02"
            assert len(merged_events) == 18, (
                f"Expected 18 events, got {len(merged_events)}"
            )

            # Check first event from file 1
            assert merged_events[0]["time"] == 0

            # Check first event from file 3 (file 2 had no EV02)
            assert merged_events[10]["time"] == 20_000_000
