# Python Tools for the DAQ-HDF5 format

A Python package for handling
[DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5)(`*.dh5`) files. 

[![Python Tests](https://github.com/cog-neurophys-lab/dh5io/actions/workflows/python-tests.yml/badge.svg)](https://github.com/cog-neurophys-lab/dh5io/actions/workflows/python-tests.yml)

- **`dhspec`** contains the specification of the DAQ-HDF5 file format as Python code.
- **`dh5io`** contains code for reading, writing and validating HDF5 files containing data
  according to the DAQ-HDF5 specfication.
- **`dh5neo` (WIP)** contains code for reading DAQ-HDF5 data into
  [Neo](https://github.com/NeuralEnsemble/python-neo) objects (e.g. for use with [Elephant](https://elephant.readthedocs.io/en/latest/index.html), [SpikeInterface](https://spikeinterface.readthedocs.io) and [ephyviewer](https://ephyviewer.readthedocs.io/)

**Design goals**

- [ ] Provide same functionality as MATLAB dhfun (read and write CONT, TRIALMAP, SPIKE, WAVELET data)
  - [X] Create and validate a DH5 file
  - [X] Read/write/validate CONT data 
  - [ ] Read/write/validate SPIKE data 
  - [ ] Read/write/validate WAVELET data 
- [ ] Provide a [Neo](https://github.com/NeuralEnsemble/python-neo) IO module to enable integration in the Neo ecosystem (Elephant, ...)
- [ ] Provide CLI tool for inspecting a DH5 file
- [ ] Provide a GUI tool for inspecting a DH5 file


Checkout Neo's developer guide: https://neo.readthedocs.io/en/stable/io_developers_guide.html

This project is still very much work-in-progress and contains nothing usable yet.


## Developer setup

- Use [uv](https://docs.astral.sh/uv)
- Setup pre-push hook for running pytest 
  ```bash
  git config --local core.hooksPath .githooks
  ```