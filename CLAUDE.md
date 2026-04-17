# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup (recommended: use [uv](https://docs.astral.sh/uv))**
```bash
uv sync --extra dev        # install with all dev dependencies
git config --local core.hooksPath .githooks  # enable pre-push test hook
```

**Running tests**
```bash
uv run pytest tests                          # run all tests
uv run pytest tests/test_create_dh5file.py  # run a single test file
uv run pytest tests -k "test_create"        # run tests matching a pattern
```

**Linting / formatting**
```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy src            # type check
```

**CLI tools (after install)**
```bash
dh5tree myfile.dh5         # display file contents as a tree
dh5merge ...               # merge DH5 files
dh5browser myfile.dh5      # interactive GUI viewer (requires dh5io[browser])
```

## Architecture

This is a monorepo-style src-layout project with four packages under `src/`:

### `dhspec` — Format specification (no I/O)
Pure Python constants, dtypes, and enums that define the DAQ-HDF5 format. No h5py dependency. Other packages import from here so the format spec stays in one place. Key files:
- `dh5file.py` — root-level HDF5 attributes (`FILEVERSION`, `BOARDS`)
- `cont.py` — CONT block constants: `CONT_PREFIX`, `DATA_DATASET_NAME`, `INDEX_DATASET_NAME`, `CONT_DTYPE_NAME`, `ContSignalType` enum, `CHANNELS_DTYPE`
- `trialmap.py` — `TRIALMAP_DATASET_DTYPE`, field names

### `dh5io` — Core I/O library
Reads, writes, and validates `.dh5` files using h5py. Public API is exported from `__init__.py`:
- `DH5File` — context-manager wrapper around `h5py.File`; main entry point for users
- `Cont` — wrapper around a `CONT{n}` h5py.Group; exposes `.data` (int16), `.calibrated_data` (float64), `.index`, `.sample_period`, etc.
- `Trialmap` — wrapper around the `TRIALMAP` structured numpy recarray
- `Wavelet` — wrapper around `WAVELET{n}` groups
- `create_dh_file()` — creates a new valid DH5 file including the shared `CONT_INDEX_ITEM` datatype
- `validate_dh5_file()` — validates structure and dtypes; raises `DH5Error` or warns with `DH5Warning` subclasses

**Two-layer API pattern**: `DH5File` provides an OO interface on top of module-level functions (e.g., `cont.get_cont_groups_from_file(h5file)`). The module-level functions take bare `h5py.File`/`h5py.Group` objects and are the building blocks for advanced use.

**Key data type detail**: `CONT.DATA` is always stored as `int16`; multiply by `Calibration` attribute (float64 per-channel array) to get volts. The `INDEX` dataset is a structured array with `time` (int64, nanoseconds) and `offset` (int64, sample offset into DATA).

### `dh5neo` — Neo integration (WIP)
Connects `dh5io` to [python-neo](https://github.com/NeuralEnsemble/python-neo) via a `RawIO`-based adapter (`DH5RawIO`) and a higher-level `DH5IO` class. Enables use with Elephant, SpikeInterface, and ephyviewer.

### `dhzio` — Zarr backend (experimental)
`DHZFolder` mirrors the DH5 structure in a Zarr store instead of HDF5.

### `dh5cli` — Command-line tools
Entry points for `dh5tree`, `dh5merge`, and `dh5browser`. The browser (`dh5browser.py`) depends on ephyviewer + PySide6 (optional dependency group `browser`).

## Optional dependency groups

Defined in `pyproject.toml`:
- `dh5io[neo]` — adds python-neo
- `dh5io[mne]` — adds MNE-Python + psutil
- `dh5io[dhzio]` — adds zarr
- `dh5io[browser]` — adds ephyviewer + PySide6 + neo
- `dh5io[test]` — adds pytest, pytest-cov, neo, dhzio, mne
- `dh5io[all]` — everything

## HDF5 string handling

HDF5 fixed-length ASCII strings require special treatment. Use helpers from `dh5io/hdf5_strings.py` (`ascii_str`, `ascii_str_array`, `decode_str_attr`) when reading or writing string attributes, rather than passing plain Python strings directly.
