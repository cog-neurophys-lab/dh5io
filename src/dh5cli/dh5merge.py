"""CLI tool for merging multiple DH5 files.

This tool merges two or more DH5 files that contain the same CONT and/or WAVELET
blocks recorded at different non-overlapping times. The DATA arrays are
concatenated and the INDEX datasets are updated to reflect the new offsets.

Supported data types:
- CONT blocks: Continuous data channels
- WAVELET blocks: Time-frequency wavelet transform data
- TRIALMAP: Trial mapping information
- EV02: Event triggers

The tool finds common blocks across all input files and merges them into a
single output file, preserving temporal ordering and updating indices accordingly.
"""

import argparse
import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import List, Optional, Set, Tuple

import h5py
import numpy as np
from tqdm import tqdm

try:
    from importlib.metadata import version as get_version

    __version__ = get_version("dh5io")
except Exception:
    __version__ = "unknown"

from dh5io.cont import Cont, create_cont_group_from_data_in_file
from dh5io.create import create_dh_file
from dh5io.dh5file import DH5File
from dh5io.event_triggers import (
    add_event_triggers_to_file,
    get_event_triggers_from_file,
)
from dh5io.hdf5_strings import write_str_attr
from dh5io.operations import add_operation_to_file
from dh5io.trialmap import add_trialmap_to_file, get_trialmap_from_file
from dh5io.wavelet import (
    Wavelet,
    create_wavelet_group_from_data_in_file,
)

logger = logging.getLogger(__name__)


def merge_dh5_files(
    input_files: List[Path],
    output_file: Path,
    cont_ids: Optional[List[int]] = None,
    overwrite: bool = False,
) -> None:
    """
    Merge multiple DH5 files into a single output file.

    This function merges CONT blocks, WAVELET blocks, TRIALMAPs, and event triggers
    from multiple input files. Only data blocks that are present in all input files
    will be merged.

    Parameters
    ----------
    input_files : List[Path]
        List of input DH5 file paths to merge
    output_file : Path
        Output file path
    cont_ids : Optional[List[int]]
        If provided, only merge these CONT block IDs. Otherwise, merge all common blocks.
    overwrite : bool
        Whether to overwrite the output file if it exists

    Raises
    ------
    ValueError
        If fewer than 2 input files provided, or if no common CONT or WAVELET blocks found
    FileNotFoundError
        If any input file does not exist
    """
    if len(input_files) < 2:
        raise ValueError("At least two input files are required for merging")

    # Verify all input files exist
    for file_path in input_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

    # Load all input files
    input_dh5_files = [DH5File(str(f), mode="r") for f in input_files]

    try:
        # Determine which CONT blocks to merge
        cont_blocks_to_merge = determine_cont_blocks_to_merge(input_dh5_files, cont_ids)

        # Determine which WAVELET blocks to merge
        wavelet_blocks_to_merge = determine_wavelet_blocks_to_merge(input_dh5_files)

        if not cont_blocks_to_merge and not wavelet_blocks_to_merge:
            raise ValueError("No common CONT or WAVELET blocks found to merge")

        if cont_blocks_to_merge:
            logger.info(f"Merging CONT blocks: {sorted(cont_blocks_to_merge)}")
        if wavelet_blocks_to_merge:
            logger.info(f"Will merge WAVELET blocks: {sorted(wavelet_blocks_to_merge)}")

        # Get boards attribute from first file
        boards = input_dh5_files[0]._file.attrs.get("BOARDS", [])

        # Create output file
        output_dh5 = create_dh_file(
            str(output_file),
            overwrite=overwrite,
            boards=list(boards) if len(boards) > 0 else [],
        )

        try:
            # Copy Operations group from first file to preserve processing history
            logger.info("Copying Operations group from first file...")
            copy_operations_from_first_file(input_dh5_files, output_dh5)

            # Merge each CONT block
            if cont_blocks_to_merge:
                for cont_id in sorted(cont_blocks_to_merge):
                    logger.info(f"Merging CONT{cont_id}...")
                    merge_cont_block(input_dh5_files, output_dh5, cont_id)

            # Merge TRIALMAP if present
            logger.info("Merging TRIALMAP...")
            merge_trialmaps(input_dh5_files, output_dh5)

            # Merge EV02 (event triggers) if present
            logger.info("Merging EV02 (event triggers)...")
            merge_event_triggers(input_dh5_files, output_dh5)

            # Merge WAVELET blocks if present
            if wavelet_blocks_to_merge:
                logger.info("Merging WAVELET blocks...")
                for wavelet_id in tqdm(
                    sorted(wavelet_blocks_to_merge),
                    desc="Merging WAVELET blocks",
                    unit="block",
                ):
                    merge_wavelet_block(input_dh5_files, output_dh5, wavelet_id)

            # Add operation record about the merge
            logger.info("Adding merge operation to file...")
            add_merge_operation(output_dh5, input_files)

            logger.info(
                f"Successfully merged {len(input_files)} files into {output_file}"
            )

        finally:
            output_dh5._file.close()

    finally:
        # Close all input files
        for dh5_file in input_dh5_files:
            dh5_file._file.close()


def determine_cont_blocks_to_merge(
    input_files: List[DH5File], cont_ids: Optional[List[int]] = None
) -> Set[int]:
    """
    Determine which CONT blocks should be merged.

    If cont_ids is provided, verify those blocks exist in all files.
    Otherwise, find the intersection of CONT block IDs across all files.

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files
    cont_ids : Optional[List[int]]
        Specific CONT block IDs to merge

    Returns
    -------
    Set[int]
        Set of CONT block IDs to merge
    """
    # Get CONT block IDs from each file
    file_cont_ids = [set(f.get_cont_group_ids()) for f in input_files]

    if cont_ids is not None:
        # Use specified CONT IDs
        requested_ids = set(cont_ids)

        # Verify all requested IDs exist in all files
        for i, ids in enumerate(file_cont_ids):
            missing = requested_ids - ids
            if missing:
                raise ValueError(
                    f"CONT blocks {sorted(missing)} not found in file {i}: "
                    f"{input_files[i]._file.filename}"
                )

        return requested_ids
    else:
        # Find intersection of all CONT block IDs
        common_ids = file_cont_ids[0]
        for ids in file_cont_ids[1:]:
            common_ids &= ids

        if not common_ids:
            # Show what's available in each file
            for i, ids in enumerate(file_cont_ids):
                logger.warning(
                    f"File {i} ({input_files[i]._file.filename}) has CONT blocks: {sorted(ids)}"
                )

        return common_ids


def merge_cont_block(
    input_files: List[DH5File], output_file: DH5File, cont_id: int
) -> None:
    """
    Merge a single CONT block from multiple files.

    This concatenates the DATA arrays and updates the INDEX dataset
    to reflect the new offsets.

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files
    output_file : DH5File
        Output DH5 file
    cont_id : int
        CONT block ID to merge
    """
    # Load CONT blocks from all files
    cont_blocks = [f.get_cont_group_by_id(cont_id) for f in input_files]

    # Validate compatibility
    validate_cont_blocks_compatible(cont_blocks, cont_id)

    # Use attributes from the first file
    first_cont = cont_blocks[0]
    sample_period = first_cont.sample_period
    calibration = first_cont.calibration
    channels = first_cont.channels

    # Handle optional attributes safely
    name = first_cont._group.attrs.get("Name", None)
    comment = first_cont._group.attrs.get("Comment", "")
    signal_type = first_cont.signal_type

    # Concatenate data and merge indices
    merged_data, merged_index = concatenate_cont_data(cont_blocks, calibration)

    # Create merged CONT block in output file
    create_cont_group_from_data_in_file(
        output_file._file,
        cont_id,
        data=merged_data,
        index=merged_index,
        sample_period_ns=np.int32(sample_period),
        calibration=calibration,
        channels=channels,
        name=name,
        comment=comment,
        signal_type=signal_type,
    )


def merge_trialmaps(input_files: List[DH5File], output_file: DH5File) -> None:
    """
    Merge TRIALMAP datasets from multiple files.

    TRIALMAPs are concatenated in order. If files have no TRIALMAP,
    this function does nothing.

    Validates that:
    - All trials have increasing timestamps (StartTime)
    - Trials in file N+1 come after trials in file N

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files
    output_file : DH5File
        Output DH5 file
    """
    # Collect trialmaps from all files
    trialmaps = []
    for i, dh5_file in enumerate(input_files):
        trialmap = get_trialmap_from_file(dh5_file._file)
        if trialmap is not None:
            trialmaps.append(trialmap)
            logger.debug(f"File {i}: Found TRIALMAP with {len(trialmap)} trials")
        else:
            logger.debug(f"File {i}: No TRIALMAP found")

    # If no trialmaps found, nothing to merge
    if not trialmaps:
        logger.info("No TRIALMAPs found in input files")
        return

    # Validate that trials have increasing timestamps within each file
    for i, trialmap in enumerate(trialmaps):
        if len(trialmap) > 0:
            start_times = trialmap["StartTime"]
            if not np.all(np.diff(start_times) > 0):
                logger.warning(
                    f"TRIALMAP in file {i}: Trial StartTime values are not strictly increasing. "
                    f"This may indicate a data quality issue."
                )

    # Validate that trialmaps are sequential across files
    for i in range(len(trialmaps) - 1):
        current_trialmap = trialmaps[i]
        next_trialmap = trialmaps[i + 1]

        if len(current_trialmap) == 0 or len(next_trialmap) == 0:
            continue  # Skip empty trialmaps

        # Get the last trial's end time from current file
        current_last_end = np.max(current_trialmap["EndTime"])

        # Get the first trial's start time from next file
        next_first_start = np.min(next_trialmap["StartTime"])

        # Check that next file's trials come after current file's trials
        if next_first_start < current_last_end:
            raise ValueError(
                f"TRIALMAP: Data is not sequential between files {i} and {i + 1}. "
                f"File {i + 1} has a trial starting at {next_first_start} ns, which overlaps with "
                f"or comes before file {i}'s last trial ending at {current_last_end} ns. "
                f"Files must contain non-overlapping, sequential trials for merging."
            )

    # Concatenate all trialmaps
    merged_trialmap_array = np.concatenate(trialmaps)

    # Convert to recarray if needed
    if not isinstance(merged_trialmap_array, np.recarray):
        merged_trialmap = np.rec.array(merged_trialmap_array)
    else:
        merged_trialmap = merged_trialmap_array

    logger.info(
        f"Merged {len(trialmaps)} TRIALMAPs ({len(merged_trialmap)} total trials)"
    )

    # Add to output file
    add_trialmap_to_file(output_file._file, merged_trialmap, replace=True)


def merge_event_triggers(input_files: List[DH5File], output_file: DH5File) -> None:
    """
    Merge EV02 (event triggers) datasets from multiple files.

    Event triggers are concatenated in order. If files have no EV02,
    this function does nothing.

    Validates that:
    - All event timestamps are increasing within each file
    - Events in file N+1 come after events in file N

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files
    output_file : DH5File
        Output DH5 file
    """
    # Collect event triggers from all files
    all_event_triggers = []
    for i, dh5_file in enumerate(input_files):
        events = get_event_triggers_from_file(dh5_file._file)
        if events is not None:
            all_event_triggers.append(events)
            logger.debug(f"File {i}: Found EV02 with {len(events)} events")
        else:
            logger.debug(f"File {i}: No EV02 found")

    # If no event triggers found, nothing to merge
    if not all_event_triggers:
        logger.info("No EV02 datasets found in input files")
        return

    # Validate that event timestamps are increasing within each file
    for i, events in enumerate(all_event_triggers):
        if len(events) > 0:
            event_times = events["time"]
            if not np.all(np.diff(event_times) >= 0):
                logger.warning(
                    f"EV02 in file {i}: Event timestamps are not strictly increasing. "
                    f"This may indicate a data quality issue."
                )

    # Validate that events are sequential across files
    for i in range(len(all_event_triggers) - 1):
        current_events = all_event_triggers[i]
        next_events = all_event_triggers[i + 1]

        if len(current_events) == 0 or len(next_events) == 0:
            continue  # Skip empty event datasets

        # Get the last event time from current file
        current_last_time = np.max(current_events["time"])

        # Get the first event time from next file
        next_first_time = np.min(next_events["time"])

        # Check that next file's events come after current file's events
        if next_first_time < current_last_time:
            raise ValueError(
                f"EV02: Data is not sequential between files {i} and {i + 1}. "
                f"File {i + 1} has an event at time {next_first_time} ns, which overlaps with "
                f"or comes before file {i}'s last event at {current_last_time} ns. "
                f"Files must contain non-overlapping, sequential events for merging."
            )

    # Concatenate all event triggers
    merged_events = np.concatenate(all_event_triggers)

    logger.info(
        f"Merged {len(all_event_triggers)} EV02 datasets ({len(merged_events)} total events)"
    )

    # Extract timestamps and event codes
    timestamps = merged_events["time"]
    event_codes = merged_events["event"]

    # Add to output file
    add_event_triggers_to_file(output_file._file, timestamps, event_codes)


def determine_wavelet_blocks_to_merge(input_files: List[DH5File]) -> Set[int]:
    """
    Determine which WAVELET blocks should be merged.

    Find the intersection of WAVELET block IDs across all files.

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files

    Returns
    -------
    Set[int]
        Set of WAVELET block IDs to merge
    """
    # Get WAVELET block IDs from each file
    file_wavelet_ids = [set(f.get_wavelet_group_ids()) for f in input_files]

    # Find intersection of all WAVELET block IDs
    if not file_wavelet_ids or not file_wavelet_ids[0]:
        return set()

    common_ids = file_wavelet_ids[0]
    for ids in file_wavelet_ids[1:]:
        common_ids &= ids

    if not common_ids:
        # Show what's available in each file
        for i, ids in enumerate(file_wavelet_ids):
            if ids:
                logger.debug(
                    f"File {i} ({input_files[i]._file.filename}) has WAVELET blocks: {sorted(ids)}"
                )

    return common_ids


def validate_wavelet_blocks_compatible(
    wavelet_blocks: List[Wavelet], wavelet_id: int
) -> None:
    """
    Validate that WAVELET blocks from different files are compatible for merging.

    All blocks must have the same:
    - Sample period
    - Number of channels
    - Number of frequencies
    - Frequency axis values

    Additionally validates that data is sequential (non-overlapping in time):
    - Data in file N+1 must come after data in file N

    Parameters
    ----------
    wavelet_blocks : List[Wavelet]
        List of WAVELET blocks to validate
    wavelet_id : int
        WAVELET block ID (for error messages)
    """
    first = wavelet_blocks[0]

    for i, wavelet in enumerate(wavelet_blocks[1:], start=1):
        # Check sample period
        if wavelet.sample_period != first.sample_period:
            raise ValueError(
                f"WAVELET{wavelet_id}: Sample period mismatch in file {i}. "
                f"Expected {first.sample_period} ns, got {wavelet.sample_period} ns"
            )

        # Check number of channels
        if wavelet.n_channels != first.n_channels:
            raise ValueError(
                f"WAVELET{wavelet_id}: Number of channels mismatch in file {i}. "
                f"Expected {first.n_channels}, got {wavelet.n_channels}"
            )

        # Check number of frequencies
        if wavelet.n_frequencies != first.n_frequencies:
            raise ValueError(
                f"WAVELET{wavelet_id}: Number of frequencies mismatch in file {i}. "
                f"Expected {first.n_frequencies}, got {wavelet.n_frequencies}"
            )

        # Check frequency axis
        if not np.allclose(wavelet.frequency_axis, first.frequency_axis):
            raise ValueError(
                f"WAVELET{wavelet_id}: Frequency axis mismatch in file {i}. "
                f"Expected {first.frequency_axis}, got {wavelet.frequency_axis}"
            )

    # Validate that data is sequential (non-overlapping in time)
    for i in range(len(wavelet_blocks) - 1):
        current_block = wavelet_blocks[i]
        next_block = wavelet_blocks[i + 1]

        # Calculate the end time of current block's data
        # End time = last sample's timestamp + (n_samples * sample_period)
        if current_block.n_samples == 0:
            continue  # Skip empty blocks

        current_index = current_block.index[:]
        if len(current_index) == 0:
            continue

        # Find the last recording region
        last_region_idx = np.argmax(current_index["time"])
        last_region_start = current_index[last_region_idx]["time"]
        last_region_offset = current_index[last_region_idx]["offset"]

        # Calculate how many samples are in this last region
        samples_in_last_region = current_block.n_samples - last_region_offset

        # Calculate end time: start of last region + duration of samples in that region
        current_end_time = last_region_start + (
            samples_in_last_region * current_block.sample_period
        )

        # Get the first timestamp from next block
        next_index = next_block.index[:]
        if len(next_index) == 0 or next_block.n_samples == 0:
            continue  # Skip empty blocks
        next_first_time = np.min(next_index["time"])

        # Check that next file's data comes after current file's data
        if next_first_time < current_end_time:
            raise ValueError(
                f"WAVELET{wavelet_id}: Data is not sequential between files {i} and {i + 1}. "
                f"File {i + 1} starts at {next_first_time} ns, which overlaps with "
                f"file {i}'s data ending at {current_end_time} ns. "
                f"Files must contain non-overlapping, sequential data for merging."
            )


def concatenate_wavelet_data(
    wavelet_blocks: List[Wavelet],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate DATA arrays and merge INDEX datasets from multiple WAVELET blocks.

    The INDEX dataset is updated so that offsets reflect the concatenated DATA array.
    Time stamps are preserved from the original recordings.

    Parameters
    ----------
    wavelet_blocks : List[Wavelet]
        List of WAVELET blocks to concatenate

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (merged_amplitude, merged_phase, merged_index)
    """
    # Collect amplitude and phase data from all blocks
    amplitude_arrays = []
    phase_arrays = []
    index_arrays = []
    data_shapes = []

    for wavelet in wavelet_blocks:
        # Get calibrated amplitude and phase for this block
        amplitude = wavelet.get_amplitude_calibrated()
        phase = wavelet.get_phase_radians()

        amplitude_arrays.append(amplitude)
        phase_arrays.append(phase)
        index_arrays.append(wavelet.index[:])

        # Shape is (N, M, F) - we only need M (samples dimension)
        data_shapes.append(amplitude.shape[1])

    # Concatenate along the samples axis (axis 1: M dimension)
    merged_amplitude = np.concatenate(amplitude_arrays, axis=1)
    merged_phase = np.concatenate(phase_arrays, axis=1)

    # Merge index arrays - need to adjust offsets
    merged_indices = []
    cumulative_samples = 0

    for index_array, n_samples in zip(index_arrays, data_shapes):
        # Copy the index array
        adjusted_index = index_array.copy()

        # Adjust offsets by the cumulative sample count
        adjusted_index["offset"] += cumulative_samples

        merged_indices.append(adjusted_index)

        # Update cumulative sample count (n_samples is M dimension)
        cumulative_samples += n_samples

    merged_index = np.concatenate(merged_indices)

    return merged_amplitude, merged_phase, merged_index


def merge_wavelet_block(
    input_files: List[DH5File], output_file: DH5File, wavelet_id: int
) -> None:
    """
    Merge a single WAVELET block from multiple files.

    This concatenates the DATA arrays and updates the INDEX dataset
    to reflect the new offsets.

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files
    output_file : DH5File
        Output DH5 file
    wavelet_id : int
        WAVELET block ID to merge
    """
    # Load WAVELET blocks from all files
    wavelet_blocks = [f.get_wavelet_group_by_id(wavelet_id) for f in input_files]

    # Filter out None values (in case some files don't have this wavelet)
    wavelet_blocks = [w for w in wavelet_blocks if w is not None]

    if not wavelet_blocks:
        logger.warning(f"WAVELET{wavelet_id}: No blocks found to merge")
        return

    # Validate compatibility
    validate_wavelet_blocks_compatible(wavelet_blocks, wavelet_id)

    # Use attributes from the first file
    first_wavelet = wavelet_blocks[0]
    sample_period = first_wavelet.sample_period
    frequency_axis = first_wavelet.frequency_axis

    # Handle optional attributes safely
    name = first_wavelet.name
    comment = first_wavelet.comment

    # Concatenate data and merge indices
    merged_amplitude, merged_phase, merged_index = concatenate_wavelet_data(
        wavelet_blocks
    )

    # Create merged WAVELET block in output file
    create_wavelet_group_from_data_in_file(
        output_file._file,
        wavelet_id,
        amplitude=merged_amplitude,
        phase=merged_phase,
        index=merged_index,
        sample_period_ns=np.int32(sample_period),
        frequency_axis=frequency_axis,
        name=name,
        comment=comment,
    )


def copy_operations_from_first_file(
    input_files: List[DH5File], output_file: DH5File
) -> None:
    """
    Copy operations from the first input file to the output file.

    This preserves the processing history from the first file in the merged output.
    Operations are copied and renumbered to come after the output file's initial
    "create_file" operation.

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files
    output_file : DH5File
        Output DH5 file
    """
    from dh5io.operations import (
        OPERATIONS_GROUP_NAME,
        get_last_operation_index,
        get_operations_group,
    )

    first_file = input_files[0]
    source_operations = get_operations_group(first_file._file)
    dest_operations = get_operations_group(output_file._file)

    if source_operations is None:
        logger.debug("No Operations group found in first file")
        return

    if dest_operations is None:
        # This shouldn't happen since create_dh_file adds an operation
        logger.warning("No Operations group in output file, creating one")
        dest_operations = output_file._file.create_group(OPERATIONS_GROUP_NAME)
        next_index = 0
    else:
        # Get the next available index in the output file
        next_index = get_last_operation_index(output_file._file) + 1

    # Copy each operation from the source, renumbering as needed
    copied_count = 0
    for op_name in sorted(source_operations.keys()):
        # Extract the operation name without the index prefix
        parts = op_name.split("_", 1)
        if len(parts) == 2:
            operation_name = parts[1]
        else:
            operation_name = op_name

        # Skip 'create_file' operations from the first file since the merged file
        # already has its own create_file operation
        if operation_name == "create_file":
            continue

        # Create new operation name with the next index
        new_op_name = f"{next_index:03}_{operation_name}"

        # Copy the operation group
        first_file._file.copy(
            f"{OPERATIONS_GROUP_NAME}/{op_name}",
            dest_operations,
            name=new_op_name,
        )

        next_index += 1
        copied_count += 1

    logger.info(
        f"Copied {copied_count} operation(s) from first file to output (renumbered starting at index {get_last_operation_index(output_file._file) - copied_count + 1})"
    )


def add_merge_operation(output_file: DH5File, input_files: List[Path]) -> None:
    """
    Add an operation record to the output file documenting the merge.

    Parameters
    ----------
    output_file : DH5File
        Output DH5 file
    input_files : List[Path]
        List of input file paths that were merged
    """
    # Create a description of the merge operation
    input_filenames = [f.name for f in input_files]

    # Add operation to file
    add_operation_to_file(
        output_file._file,
        new_operation_group_name="merge_files",
        tool="dh5merge (dh5io)",
    )

    # Get the last operation group to add custom attributes
    from dh5io.operations import get_last_operation_index, get_operations_group

    operations_group = get_operations_group(output_file._file)
    if operations_group is not None:
        last_index = get_last_operation_index(output_file._file)
        if last_index is not None:
            operation_group_name = f"{last_index:03}_merge_files"
            operation_group = operations_group[operation_group_name]

            # Add custom attributes about the merge
            write_str_attr(operation_group, "MergedFiles", ", ".join(input_filenames))
            operation_group.attrs["NumberOfFiles"] = len(input_files)

            logger.debug(f"Added merge operation with {len(input_files)} source files")


def validate_cont_blocks_compatible(cont_blocks: List[Cont], cont_id: int) -> None:
    """
    Validate that CONT blocks from different files are compatible for merging.

    All blocks must have the same:
    - Sample period
    - Number of channels
    - Calibration (if present)
    - Channel configuration

    Additionally validates that data is sequential (non-overlapping in time):
    - Data in file N+1 must come after data in file N

    Parameters
    ----------
    cont_blocks : List[Cont]
        List of CONT blocks to validate
    cont_id : int
        CONT block ID (for error messages)
    """
    first = cont_blocks[0]

    for i, cont in enumerate(cont_blocks[1:], start=1):
        # Check sample period
        if cont.sample_period != first.sample_period:
            raise ValueError(
                f"CONT{cont_id}: Sample period mismatch in file {i}. "
                f"Expected {first.sample_period} ns, got {cont.sample_period} ns"
            )

        # Check number of channels
        if cont.n_channels != first.n_channels:
            raise ValueError(
                f"CONT{cont_id}: Number of channels mismatch in file {i}. "
                f"Expected {first.n_channels}, got {cont.n_channels}"
            )

        # Check calibration
        first_calib = first.calibration
        cont_calib = cont.calibration

        if (first_calib is None) != (cont_calib is None):
            raise ValueError(
                f"CONT{cont_id}: Calibration presence mismatch in file {i}"
            )

        if first_calib is not None and cont_calib is not None:
            if not np.allclose(first_calib, cont_calib):
                logger.info(
                    f"CONT{cont_id}: Calibration values differ in file {i}. "
                    f"Will perform conversion to preserve signal values."
                )

    # Validate that data is sequential (non-overlapping in time)
    for i in range(len(cont_blocks) - 1):
        current_block = cont_blocks[i]
        next_block = cont_blocks[i + 1]

        # Calculate the end time of current block's data
        # End time = last sample's timestamp + (n_samples * sample_period)
        if current_block.n_samples == 0:
            continue  # Skip empty blocks

        current_index = current_block.index[:]
        if len(current_index) == 0:
            continue

        # Find the last recording region
        last_region_idx = np.argmax(current_index["time"])
        last_region_start = current_index[last_region_idx]["time"]
        last_region_offset = current_index[last_region_idx]["offset"]

        # Calculate how many samples are in this last region
        samples_in_last_region = current_block.n_samples - last_region_offset

        # Calculate end time: start of last region + duration of samples in that region
        current_end_time = last_region_start + (
            samples_in_last_region * current_block.sample_period
        )

        # Get the first timestamp from next block
        next_index = next_block.index[:]
        if len(next_index) == 0 or next_block.n_samples == 0:
            continue  # Skip empty blocks
        next_first_time = np.min(next_index["time"])

        # Check that next file's data comes after current file's data
        if next_first_time < current_end_time:
            raise ValueError(
                f"CONT{cont_id}: Data is not sequential between files {i} and {i + 1}. "
                f"File {i + 1} starts at {next_first_time} ns, which overlaps with "
                f"file {i}'s data ending at {current_end_time} ns. "
                f"Files must contain non-overlapping, sequential data for merging."
            )


def merge_index_arrays_with_shapes(
    index_arrays: List[np.ndarray], data_shapes: List[tuple]
) -> np.ndarray:
    """
    Merge multiple INDEX arrays with knowledge of data array shapes.

    Parameters
    ----------
    index_arrays : List[np.ndarray]
        List of INDEX arrays to merge
    data_shapes : List[tuple]
        List of data array shapes (nSamples, nChannels) for CONT data
        or other shapes for different data types

    Returns
    -------
    np.ndarray
        Merged INDEX array with updated offsets
    """
    merged_indices = []
    cumulative_samples = 0

    for index_array, shape in zip(index_arrays, data_shapes):
        # Copy the index array
        adjusted_index = index_array.copy()

        # Adjust offsets by the cumulative sample count
        adjusted_index["offset"] += cumulative_samples

        merged_indices.append(adjusted_index)

        # Update cumulative sample count (nSamples is shape[0])
        cumulative_samples += shape[0]

    return np.concatenate(merged_indices)


# Update concatenate_cont_data to use the improved version
def concatenate_cont_data(
    cont_blocks: List[Cont], output_calibration: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate DATA arrays and merge INDEX datasets from multiple CONT blocks.

    The INDEX dataset is updated so that offsets reflect the concatenated DATA array.
    Time stamps are preserved from the original recordings.

    If calibration values differ between files, this function:
    1. Converts raw int16 data to float64 using each file's calibration
    2. Concatenates the calibrated float64 arrays
    3. Converts back to int16 using the output calibration value

    Parameters
    ----------
    cont_blocks : List[Cont]
        List of CONT blocks to concatenate
    output_calibration : np.ndarray | None
        Calibration values to use for the output file. If None, no calibration
        conversion is performed (data is concatenated as raw int16).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Merged data array (int16) and merged index array
    """
    # Check if we need to handle calibration conversion
    calibrations = [cont.calibration for cont in cont_blocks]
    needs_conversion = False

    # Only perform conversion if calibrations exist and differ
    if output_calibration is not None and all(cal is not None for cal in calibrations):
        # Check if any calibration differs from others
        for i, cal in enumerate(calibrations):
            if i > 0 and not np.allclose(cal, calibrations[0]):
                needs_conversion = True
                break

    if needs_conversion:
        logger.info(
            "Calibration values differ between files. "
            "Performing data conversion to preserve signal values during merge."
        )
        # Convert each block to calibrated float64, concatenate, then convert back
        calibrated_arrays = []
        for cont in cont_blocks:
            # Convert int16 to float64 using the source file's calibration
            calibrated_data = cont.data.astype(np.float64) * cont.calibration
            calibrated_arrays.append(calibrated_data)

        # Concatenate calibrated float64 arrays
        merged_calibrated = np.concatenate(calibrated_arrays, axis=0)

        # Convert back to int16 using output calibration
        merged_data_float = merged_calibrated / output_calibration

        # Clip to int16 range and convert
        merged_data_float = np.clip(merged_data_float, -32768, 32767)
        merged_data = merged_data_float.astype(np.int16)

        data_shapes = [arr.shape for arr in calibrated_arrays]
    else:
        # No calibration conversion needed - use raw int16 data directly
        logger.debug("Calibration values match. Concatenating raw int16 data.")
        data_arrays = [cont.data for cont in cont_blocks]
        merged_data = np.concatenate(data_arrays, axis=0)
        data_shapes = [arr.shape for arr in data_arrays]

    # Merge index arrays with knowledge of data shapes
    merged_index = merge_index_arrays_with_shapes(
        [cont.index for cont in cont_blocks], data_shapes
    )

    return merged_data, merged_index


def suggest_merged_filename(input_files: List[Path]) -> Optional[Path]:
    """
    Suggest a merged filename based on the common prefix of input filenames.

    Parameters
    ----------
    input_files : List[Path]
        List of input file paths

    Returns
    -------
    Optional[Path]
        Suggested output filename with "_merged" suffix, or None if no overlap

    Examples
    --------
    >>> suggest_merged_filename([Path("session1_day1.dh5"), Path("session1_day2.dh5")])
    Path("session1_merged.dh5")
    >>> suggest_merged_filename([Path("file1.dh5"), Path("data2.dh5")])
    None
    """
    if len(input_files) < 2:
        return None

    # Get basenames without extensions
    basenames = [f.stem for f in input_files]

    # Find the longest common prefix
    if not basenames:
        return None

    # Start with the first basename as reference
    common_prefix = basenames[0]

    # Compare with all other basenames
    for basename in basenames[1:]:
        # Find common prefix between current common_prefix and this basename
        new_prefix = ""
        for i, (c1, c2) in enumerate(zip(common_prefix, basename)):
            if c1 == c2:
                new_prefix += c1
            else:
                break
        common_prefix = new_prefix

        # Early exit if no common prefix
        if not common_prefix:
            return None

    # Remove trailing underscores, hyphens, or spaces from the common prefix
    common_prefix = common_prefix.rstrip("_- ")

    # Check if we have a meaningful overlap (at least 2 characters)
    if len(common_prefix) < 2:
        return None

    # Use the directory of the first input file
    output_dir = input_files[0].parent

    # Create the suggested filename
    suggested_name = f"{common_prefix}_merged.dh5"
    return output_dir / suggested_name


def select_files_gui() -> tuple[List[Path], Path, bool]:
    """
    Open a GUI to select input files, output file, and merge options.

    Returns
    -------
    tuple[List[Path], Path, bool]
        A tuple of (input_files, output_file, overwrite)
        Returns ([], None, False) if user cancels
    """
    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # Select input files
    input_files_str = filedialog.askopenfilenames(
        title="Select DH5 files to merge (at least 2)",
        filetypes=[("DH5 files", "*.dh5"), ("All files", "*.*")],
    )

    if not input_files_str or len(input_files_str) < 2:
        messagebox.showwarning(
            "Invalid Selection", "Please select at least 2 DH5 files to merge."
        )
        root.destroy()
        return [], None, False

    input_files = [Path(f) for f in input_files_str]

    # Suggest a merged filename based on overlap
    suggested_filename = suggest_merged_filename(input_files)

    if suggested_filename is None:
        # No overlap - abort with message
        messagebox.showerror(
            "No Common Prefix",
            "Cannot suggest a merged filename: input files have no common prefix.\n"
            "Please ensure your files follow a consistent naming pattern.",
        )
        root.destroy()
        return [], None, False

    # Select output file with suggested name as default
    output_file_str = filedialog.asksaveasfilename(
        title="Select output file location",
        defaultextension=".dh5",
        filetypes=[("DH5 files", "*.dh5"), ("All files", "*.*")],
        initialfile=suggested_filename.name,
        initialdir=suggested_filename.parent,
    )

    if not output_file_str:
        messagebox.showinfo("Cancelled", "Merge operation cancelled.")
        root.destroy()
        return [], None, False

    output_file = Path(output_file_str)

    # Check if output file exists and ask for overwrite confirmation
    overwrite = False
    if output_file.exists():
        result = messagebox.askyesno(
            "File Exists",
            f"The file {output_file.name} already exists.\nDo you want to overwrite it?",
        )
        if not result:
            messagebox.showinfo("Cancelled", "Merge operation cancelled.")
            root.destroy()
            return [], None, False
        overwrite = True

    root.destroy()
    return input_files, output_file, overwrite


def main():
    """Main entry point for the dh5merge CLI tool."""
    # Check if running without arguments (GUI mode)
    if len(sys.argv) == 1:
        # GUI mode
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        input_files, output_file, overwrite = select_files_gui()

        if not input_files or output_file is None:
            # User cancelled
            sys.exit(0)

        try:
            # Show message that merge is starting
            print(f"dh5io version: {__version__}")
            print(f"Merging {len(input_files)} files...")
            for i, f in enumerate(input_files, 1):
                print(f"  {i}. {f.name}")
            print(f"Output: {output_file}")
            print()

            # Perform merge
            merge_dh5_files(
                input_files=input_files,
                output_file=output_file,
                cont_ids=None,
                overwrite=overwrite,
            )

            # Show success message box
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(
                "Success",
                f"Successfully merged {len(input_files)} files into:\n{output_file.name}",
            )
            root.destroy()

        except Exception as e:
            logger.error(f"Error merging files: {e}")

            # Show error message box
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", f"Error merging files:\n{str(e)}")
            root.destroy()

            sys.exit(1)
    else:
        # CLI mode with arguments
        parser = argparse.ArgumentParser(
            description="Merge multiple DH5 files containing the same CONT blocks, WAVELET blocks, and other data recorded at different times.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Open GUI file selector
  dh5merge

  # Merge all common CONT and WAVELET blocks from multiple files
  dh5merge file1.dh5 file2.dh5 file3.dh5 -o merged.dh5

  # Merge only specific CONT blocks (WAVELET blocks still merged if common)
  dh5merge file1.dh5 file2.dh5 -o merged.dh5 --cont-ids 0 1 2

  # Overwrite existing output file
  dh5merge file1.dh5 file2.dh5 -o merged.dh5 --overwrite

  # Enable verbose logging
  dh5merge file1.dh5 file2.dh5 -o merged.dh5 -v

Note: The tool merges CONT blocks, WAVELET blocks, TRIALMAPs, and event triggers
that are common to all input files.
            """,
        )

        parser.add_argument(
            "input_files",
            type=str,
            nargs="+",
            help="Input DH5 files to merge (at least 2 required)",
        )

        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=False,
            help="Output DH5 file path (if not specified, will auto-suggest based on common filename prefix)",
        )

        parser.add_argument(
            "--cont-ids",
            type=int,
            nargs="+",
            help="CONT block IDs to merge (if not specified, all common blocks will be merged)",
        )

        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite output file if it exists",
        )

        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable verbose logging"
        )

        args = parser.parse_args()

        # Setup logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        try:
            # Print version
            print(f"dh5io version: {__version__}")

            # Convert input file paths
            input_files = [Path(f) for f in args.input_files]

            # Determine output file
            if args.output:
                output_file = Path(args.output)
            else:
                # No output specified - suggest based on overlap
                suggested_output = suggest_merged_filename(input_files)
                if suggested_output is None:
                    logger.error(
                        "Cannot suggest a merged filename: input files have no common prefix.\n"
                        "Please specify an output file with -o/--output or ensure your files follow a consistent naming pattern."
                    )
                    sys.exit(1)
                output_file = suggested_output
                logger.info(f"Suggested output filename: {output_file}")

            # Perform merge
            merge_dh5_files(
                input_files=input_files,
                output_file=output_file,
                cont_ids=args.cont_ids,
                overwrite=args.overwrite,
            )

        except Exception as e:
            logger.error(f"Error merging files: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
