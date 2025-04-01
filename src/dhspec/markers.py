"""
Time markers

Markers are used to describe some important points in time during the recording session,
other than trial starts and ends.

Each marker has a symbolic name and a set of times of occurrence.

In DAQ-HDF files, markers are stored under the '/Markers' group. This group can contain
multiple datasets. Each of these datasets is named after the marker symbolic name, and
stores a one-dimensional array of 64-bit integers. These are timestamps in nanoseconds.

If no markers are specified for a DAQ-HDF file, the '/Markers' group may be absent
altogether.
"""

import numpy as np

MARKERS_GROUP_NAME = "Markers"
MARKERS_DATASET_DTYPE = np.int64
