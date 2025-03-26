import h5py
from dh5io.errors import DH5Error, DH5Warning
import warnings

TRIALMAP_DATASET_NAME = "TRIALMAP"


def validate_trialmap(file: h5py.File):
    # check for TRIALMAP dataset
    if TRIALMAP_DATASET_NAME not in file:
        warnings.warn(
            message=f"TRIALMAP dataset is missing from {file}", category=DH5Warning
        )
        return
    validate_trialmap_dataset(file[TRIALMAP_DATASET_NAME])


def validate_trialmap_dataset(trialmap: h5py.Dataset) -> None:
    # trialmap must be a compound dataset with fields 'TrialNo', 'StimNo', 'Outcome', 'StartTime', 'EndTime'
    if not isinstance(trialmap, h5py.Dataset) or trialmap.dtype.names != (
        "TrialNo",
        "StimNo",
        "Outcome",
        "StartTime",
        "EndTime",
    ):
        raise DH5Error(
            f"TRIALMAP dataset is not a named data type with fields 'TrialNo', 'StimNo', 'Outcome', 'StartTime', 'EndTime': {trialmap.dtype}"
        )
