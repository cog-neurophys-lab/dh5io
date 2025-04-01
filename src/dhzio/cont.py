import zarr
import numpy.typing as npt
import numpy as np
from dhzio.errors import DHZError

CONT_PREFIX = "CONT"
DATA_DATASET_NAME = "DATA"
INDEX_DATASET_NAME = "INDEX"
CONT_DTYPE_NAME = "CONT_INDEX_ITEM"

CONT_DTYPE = np.dtype([("time", np.int64), ("offset", np.int64)])


def cont_name_from_id(id: int) -> str:
    return f"{CONT_PREFIX}{id}"


def enumerate_cont_groups(root: zarr.Group) -> list[int]:
    return [
        int(name[len(CONT_PREFIX) :])
        for name in root.group_keys()
        if name.startswith(CONT_PREFIX)
    ]


def create_empty_cont_group(
    root: zarr.Group,
    cont_group_id: int | None,
    nSamples: int,
    nChannels: int,
    sample_period_ns: int,
    n_index_items: int = 1,
    # numpy array with dtype=np.float64 of length nChannels describing calibration
    calibration: npt.NDArray[np.float64] | None = None,
    # numpy array with dtype=CHANNELS_DTYPE of length nChannels describing channels
    channels: np.ndarray | None = None,
    name: str | None = None,
    comment: str | None = None,
) -> zarr.Group:
    existing_cont_ids = enumerate_cont_groups(root)

    # fail if CONT group already exists
    if cont_group_id in existing_cont_ids:
        raise DHZError(f"CONT{cont_group_id} already exists in the folder.")

    # if cont_group_id is None:
    # cont_group_id = np.max(np.array(existing_cont_ids)) + 1

    cont_group = root.create_group(cont_name_from_id(cont_group_id))

    cont_group.create_array(
        DATA_DATASET_NAME, shape=(nSamples, nChannels), dtype=np.int16
    )
    cont_group.create_array(INDEX_DATASET_NAME, shape=(n_index_items,), dtype=np.int64)

    cont_group.attrs["SamplePeriod"] = np.int32(sample_period_ns)

    # optional attributes
    if calibration is not None:
        cont_group.attrs["Calibration"] = calibration

    if channels is not None:
        cont_group.attrs["Channels"] = channels

    # set name attribute
    if name is not None:
        cont_group.attrs["Name"] = name
    else:
        cont_group.attrs["Name"] = f"CONT{cont_group_id}"

    # set comment attribute
    if comment is not None:
        cont_group.attrs["Comment"] = comment
    else:
        cont_group.attrs["Comment"] = ""

    return cont_group
