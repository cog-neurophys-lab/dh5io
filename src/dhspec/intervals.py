"""
Time intervals

Intervals are similar to markers, however they describe not single
points in time, but ranges in time, or intervals.

Each interval has a symbolic name and a set of time ranges of its
occurrence.

In DAQ-HDF files, intervals are stored under the `/Intervals` group.
This group can contain multiple datasets. Each of these datasets is
named after the interval's symbolic name, and stores a one-dimensional
array of structures:

| Offset | Name | Type |
|-----|-----------|-------|
| 0   | StartTime | `int64` |
| 8   | EndTime   | `int64` |

StartTime and EndTime are specified in nanoseconds, and they tell the
starting and ending points in time for each occurrence of a given
interval.

Composition of this structure is stored as a shared HDF5 datatype in the
`/Intervals` group under the name `INTERVAL`. So, the `/Intervals` group
contains datasets as well as one shared datatype.

If no intervals are specified for a DAQ-HDF file, the `/Intervals` group
can be absent as well as the `INTERVAL` shared datatype"
"""

import numpy as np

INTERVAL_GROUP_NAME = "Intervals"
INTERVAL_DATASET_DTYPE = np.dtype(
    [
        ("start_time", np.int64),
        ("end_time", np.int64),
    ]
)
