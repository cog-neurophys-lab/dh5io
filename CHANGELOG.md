# Changelog for dh5io

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-04-17

### Added

- Added `add_write_trialmap_operation()` helper to write an Operations entry for TRIALMAP for backwards compatibility with BrainBox.

## [0.4.0] - 2026-02-23

### Added

- **MNE integration** (`dh5io.dh5mne`): New module for reading DH5 files as MNE-Python `Raw` objects
  - `MneRawDH5`: lazy-loading Raw object backed by a DH5 file
  - `read_raw_dh5`: read a DH5 file as a single `MneRawDH5`, selecting CONT groups by ID
  - `read_raw_dh5_per_sfreq`: read one `MneRawDH5` per unique sampling rate, grouping all matching CONT blocks together
  - `read_raw_dh5_per_cont`: read one `MneRawDH5` per CONT group
  - `annotations_from_dh5`: build MNE `Annotations` from TRIALMAP and EV02 events
  - `epochs_from_dh5`: create MNE `Epochs` from a `MneRawDH5` using TRIALMAP annotations
  - Acquisition gaps between CONT regions are marked as `BAD_ACQ_SKIP` annotations
  - `require_matching_sampling_rate` kwarg to control error vs skip behaviour on mixed-rate CONT groups
- **`DH5File.is_continuous()`** and free function `is_continuous(fname)`: return `True` when all CONT blocks have a single region (no acquisition gaps)
- **`DH5File.cont_blocks_start_simultaneously()`** and free function `cont_blocks_start_simultaneously(fname)`: return `True` when all CONT blocks share the same first-region start timestamp
- **`DH5File.get_cont_groups_by_sfreq()`**: return CONT groups grouped by sampling rate
- **`DH5File.get_cont_groups_by_ids(ids)`**: return CONT groups for a given list of IDs, raising `DH5Error` on missing IDs
- **Specific `DH5Warning` subclasses** for finer-grained warning filtering:
  - `DH5CalibrationMissingWarning(cont_id=...)` — missing `Calibration` attribute on a CONT block
  - `kDH5ChannelsMissingWarning(cont_id=...)` — missing `Channels` attribute on a CONT block
  - `DH5DataTypeConversionWarning` — CONT data silently converted to `int16` on write
  - `DH5OperationIndexWarning` — malformed or non-sequential operation index
  - `DH5DiscontinuousRegionsWarning` — file contains multiple discontinuous CONT regions
  - `DH5SampleCountMismatchWarning` — selected CONT blocks have different sample counts
  - `DH5SampleRateMismatchWarning` — CONT blocks with non-matching sampling rates were skipped
- **`DH5File.__str__`** improvements: CONT groups are now displayed grouped by sampling rate, with per-block channel count, sample count and region count; file-level `[continuous/discontinuous, simultaneous start]` flags shown at the section header
- **dhfun compatibility**: How dh5io writes string attributes to the DH5 file has been changed to be maximally compatible with
  the MATLAB tool dhfun, which uses a very old HDF5 library. In particular, this fixes errors when reading the output of dh5merge
  with dhfun (version 1).

### Fixed

- `DH5File.get_trialmap()` no longer crashes when the file contains no TRIALMAP dataset (now correctly returns `None`)

## [0.3.0] - 2026-02-05

### Added

- **Wavelet support**: Added support for reading and writing WAVELET blocks with corresponding tests
- **dh5merge tool**: New command-line tool for merging multiple DH5 files
  - Support for merging WAVELET, TRIALMAP, and EV02 blocks
  - GUI selector for choosing files to merge
  - Auto-suggest output filename based on common prefix of input files
  - Proper INDEX handling and DATA concatenation for all supported block types
  - Preserve Operations from first file on merge
  - Handle differing calibrations when merging CONT
- **dh5browser tool**: New GUI tool for browsing and visualizing DH5 files
  - Scrolling through continuous data with segment annotations
  - Trial info widget and segment annotations
- **dh5neo implementation**: Added README and tests for dh5neo subpackage to load data from DH5 files using the NEO data model.
- **Specification documentation**: Added formal DH5 file format specification (revision 3.1)

## [0.2.1] - 2025-08-21

### Fixed

- Fixed wrong package name in version getter
- Fixed broken import in tests
- Fixed trialmap test

## [0.2.0] - 2025-08-21

Initial release on PyPI.
