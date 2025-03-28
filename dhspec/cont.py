from enum import Enum
import numpy as np
import numpy.typing as npt

# specification
CONT_PREFIX = "CONT"
DATA_DATASET_NAME = "DATA"
INDEX_DATASET_NAME = "INDEX"
CONT_DTYPE_NAME = "CONT_INDEX_ITEM"
INDEX_DTYPE = np.dtype([("time", np.int64), ("offset", np.int64)])


CalibrationType = npt.NDArray[np.float64]


class ContSignalType(Enum):
    LFP = "LFP"
    MUA = "MUA/ESA"
    ANALOG = "ANALOG"
    CSD = "CSD"


CHANNELS_DTYPE = np.dtype(
    [
        ("GlobalChanNumber", np.int16),
        ("BoardChanNo", np.int16),
        ("ADCBitWidth", np.int16),
        ("MaxVoltageRange", np.float32),
        ("MinVoltageRange", np.float32),
        ("AmplifChan0", np.float32),
    ]
)


def create_channel_info(
    GlobalChanNumber: int,
    BoardChanNo: int,
    ADCBitWidth: int,
    MaxVoltageRange: float,
    MinVoltageRange: float,
    AmplifChan0: float,
) -> np.recarray:
    return np.rec.array(
        (
            GlobalChanNumber,
            BoardChanNo,
            ADCBitWidth,
            MaxVoltageRange,
            MinVoltageRange,
            AmplifChan0,
        ),
        dtype=CHANNELS_DTYPE,
    )


def create_empty_index_array(n_index_items: int) -> np.ndarray:
    return np.zeros(n_index_items, dtype=INDEX_DTYPE)


def cont_name_from_id(id: int) -> str:
    return f"{CONT_PREFIX}{id}"


def cont_id_from_name(name: str) -> int:
    return int(name.lstrip("/").lstrip(CONT_PREFIX))


CHANNELS_DTYPE = np.dtype(
    [
        ("GlobalChanNumber", np.int16),
        ("BoardChanNo", np.int16),
        ("ADCBitWidth", np.int16),
        ("MaxVoltageRange", np.float32),
        ("MinVoltageRange", np.float32),
        ("AmplifChan0", np.float32),
    ]
)
