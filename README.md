# Python Tools for the DAQ-HDF5 format

A Python package for handling
[DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5)(`*.dh5`) files. The DH5 format is
a hierarchical data format based on [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
designed for storing and sharing neurophysiology data, used in the Brain Research Institute
of the University of Bremen since 2005.

[![Python Tests](https://github.com/cog-neurophys-lab/dh5io/actions/workflows/python-tests.yml/badge.svg)](https://github.com/cog-neurophys-lab/dh5io/actions/workflows/python-tests.yml)

- **`dhspec`** contains the specification of the DAQ-HDF5 file format as Python code.
- **`dh5io`** contains code for reading, writing and validating HDF5 files containing data
  according to the DAQ-HDF5 specfication.
- **`dh5neo` (WIP)** contains code for reading DAQ-HDF5 data into
  [Neo](https://github.com/NeuralEnsemble/python-neo) objects (e.g. for use with [Elephant](https://elephant.readthedocs.io/en/latest/index.html), [SpikeInterface](https://spikeinterface.readthedocs.io) and [ephyviewer](https://ephyviewer.readthedocs.io/)

## Getting started

### Installation

Install the package using uv (recommended):

```bash
uv pip install dh5io
```

Or with pip:

```bash
pip install dh5io
```

### Reading and writing from and into DH5 files

```python
from dh5io.dh5file import DH5File

with DH5File(example_filename, "r") as dh5:
    # inspect file content
    print(dh5)

    cont = dh5.get_cont_group_by_id(1)  # Get CONT group with id 1
    print(cont)

    trialmap = dh5.get_trialmap()
    print(trialmap)
```

```
  DAQ-HDF5 File (version 2) <example_filename> containing:
      ├───CONT Groups (7):
      │   ├─── CONT1
      │   ├─── CONT60
      │   ├─── CONT61
      │   ├─── CONT62
      │   ├─── CONT63
      │   ├─── CONT64
      │   └─── CONT1001
      ├───SPIKE Groups (1):
      │   └─── SPIKE0
      ├─── 10460 Events
      └─── 385 Trials in TRIALMAP

  /CONT1 in <example_filename>
      ├─── id: 1
      ├─── name:
      ├─── comment:
      ├─── sample_period: 1000000 ns (1000.0 Hz)
      ├─── n_channels: 1
      ├─── n_samples: 1443184
      ├─── duration: 3021.76 s
      ├─── n_regions: 385
      ├─── signal_type: None
      ├─── calibration: [1.0172526e-07]
      ├─── data: (1443184, 1)
      └─── index: (385,)
```

This example shows how to open a DH5 file, inspect its content, and retrieve a specific CONT
group. The `DH5File` class provides methods for accessing the various groups and datasets
within the file. The `Cont`, `Spike` (coming in next versions) and `Trialmap` classes
provide convenient wrappers for working with these raw HDF5 groups and datasets. The
corresponding [h5py](https://docs.h5py.org/en/stable/index.html) classes can be accessed
directly for lower-level operations using the `_file`, `_group` and `_dataset` attributes
(e.g. `cont._group` or `cont.data._dataset`).

As an alternative to the object-oriented approach using `DH5File`, you can use the
functional API provided by the library. This API offers a set of functions for reading and
writing data to DH5 files without the need to create file objects. These functions in the
respective modules (`h5io.cont`, `h5io.spike`, etc.) use the
[h5py](https://docs.h5py.org/en/stable/index.html) classes as input and output. This is the
recommended way if you are familiar with HDF5 and the specification of the DH5 format.
