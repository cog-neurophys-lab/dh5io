DHZIO - Zarr implementation of DAQ-HDF 
--------------------------------------

This is an experiment that maps the DAQ-HDF specification onto a
[Zarr](https://zarr.dev) implementation. This basically maps
- arrays onto binary files
- attributes into metadata json files.