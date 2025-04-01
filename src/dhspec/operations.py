import numpy as np
import datetime

DATE_DTYPE = np.dtype(
    [
        ("Year", np.int64),
        ("Month", np.int8),
        ("Day", np.int8),
        ("Hour", np.int8),
        ("Minute", np.int8),
        ("Second", np.int8),
    ]
)


def datetime_to_date_array(dt: datetime.datetime) -> np.ndarray:
    """Convert a datetime object to a date array."""
    return np.array(
        (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second), dtype=DATE_DTYPE
    )


OPERATIONS_GROUP_NAME = "Operations"
OPERATIONS_DATE_NAME = "Date"
OPERATIONS_OPERATOR_NAME_NAME = "Operator name"
OPERATIONS_TOOL_NAME = "Tool"
OPERATIONS_ORIGINAL_FILENAME_NAME = "Original filename"
