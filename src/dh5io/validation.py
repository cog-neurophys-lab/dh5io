import logging
import pathlib
from dataclasses import dataclass
from warnings import warn

import h5py

from dh5io.cont import (
    get_cont_groups_from_file,
    validate_cont_dtype,
    validate_cont_group,
)
from dh5io.errors import DH5Error, DH5Warning
from dh5io.event_triggers import validate_event_triggers
from dh5io.operations import validate_operations
from dh5io.trialmap import validate_trialmap
from dhspec.dh5file import BOARDS_ATTRIBUTE_NAME, FILEVERSION_ATTRIBUTE_NAME

logger = logging.getLogger(__name__)


@dataclass
class ValidationOptions:
    validate_cont: bool = True
    validate_trialmap: bool = True
    validate_event_triggers: bool = True
    validate_operations: bool = True


def validate_dh5_file(
    file: str | pathlib.Path | h5py.File,
    throw=True,
    options: ValidationOptions = ValidationOptions(),
) -> None | str:
    """Validate if the given file is a valid DAQ-HDF5 file.

    This function checks if the file has the required attributes and groups.
    """

    try:
        if isinstance(file, (str, pathlib.Path)):
            file = h5py.File(file, "r")

        if not isinstance(file, (str, pathlib.Path, h5py.File)):
            raise TypeError("filename must be a str, pathlib.Path or h5py.File")

        if not isinstance(file, h5py.File):
            raise DH5Error("Not a valid HDF5 file")

        if file.attrs.get(FILEVERSION_ATTRIBUTE_NAME) is None:
            raise DH5Error(f"{FILEVERSION_ATTRIBUTE_NAME} attribute is missing")

        if file.attrs.get(BOARDS_ATTRIBUTE_NAME) is None:
            warn(f"{BOARDS_ATTRIBUTE_NAME} attribute is missing", category=DH5Warning)

        validate_cont_dtype(file)

        # check for CONT groups
        if options.validate_cont:
            cont_groups = get_cont_groups_from_file(file)
            for cont_group in cont_groups:
                validate_cont_group(cont_group)

        if options.validate_event_triggers:
            validate_event_triggers(file)

        if options.validate_trialmap:
            validate_trialmap(file)

        if options.validate_operations:
            validate_operations(file)

        return None

    except DH5Error as e:
        logger.exception(e)
        if throw:
            raise e
        else:
            return str(e)
