class DH5Error(Exception):
    pass


class DH5Warning(Warning):
    pass


class DH5CalibrationMissingWarning(DH5Warning):
    """Raised when a CONT block is missing the Calibration attribute."""

    def __init__(self, message: str, cont_id: int | str | None = None):
        super().__init__(message)
        self.cont_id = cont_id


class DH5ChannelsMissingWarning(DH5Warning):
    """Raised when a CONT block is missing the Channels attribute."""

    def __init__(self, message: str, cont_id: int | str | None = None):
        super().__init__(message)
        self.cont_id = cont_id


class DH5DataTypeConversionWarning(DH5Warning):
    """Raised when CONT data is silently converted to int16 on write."""
    pass


class DH5OperationIndexWarning(DH5Warning):
    """Raised when an operation index is malformed or non-sequential."""
    pass


class DH5DiscontinuousRegionsWarning(DH5Warning):
    """Raised when a DH5 file contains multiple discontinuous CONT regions."""
    pass


class DH5SampleCountMismatchWarning(DH5Warning):
    """Raised when selected CONT blocks have different sample counts."""
    pass


class DH5SampleRateMismatchWarning(DH5Warning):
    """Raised when CONT blocks with non-matching sampling rates are skipped."""
    pass
