import numpy

TRIALMAP_DATASET_NAME = "TRIALMAP"
TRIALMAP_DATASET_DTYPE = numpy.dtype(
    [
        ("TrialNo", "<i4"),
        ("StimNo", "<i4"),
        ("Outcome", "<i4"),
        ("StartTime", "<i8"),
        ("EndTime", "<i8"),
    ]
)
