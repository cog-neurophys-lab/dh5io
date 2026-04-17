import datetime

import h5py
import numpy as np
import pytest

from dh5io.errors import DH5Error
from dh5io.trialmap import (
    DEFAULT_OUTCOME_CODES,
    WRITE_TRIALMAP_OPERATION_NAME,
    WRITE_TRIALMAP_TOOL_NAME,
    Trialmap,
    TrialOutcome,
    add_trialmap_to_file,
    add_write_trialmap_operation,
    get_trialmap_from_file,
    validate_trialmap,
    validate_trialmap_dataset,
)
from dhspec.trialmap import TRIALMAP_DATASET_DTYPE, TRIALMAP_DATASET_NAME


@pytest.fixture
def mock_h5_file(tmp_path):
    file_path = tmp_path / "test_file.h5"
    with h5py.File(file_path, "w") as f:
        yield f


@pytest.fixture
def valid_trialmap():
    return np.rec.array(
        [
            (1, 101, 0, 1000000000, 2000000000),
            (2, 102, 1, 2000000000, 3000000000),
        ],
        dtype=TRIALMAP_DATASET_DTYPE,
    )


def test_add_trialmap_to_file(mock_h5_file, valid_trialmap):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    assert TRIALMAP_DATASET_NAME in mock_h5_file
    dataset = mock_h5_file[TRIALMAP_DATASET_NAME]
    assert dataset.dtype == TRIALMAP_DATASET_DTYPE
    assert len(dataset) == len(valid_trialmap)
    assert np.array_equal(np.array(dataset), valid_trialmap)


def test_add_trialmap_to_file_invalid_dtype(mock_h5_file):
    invalid_trialmap = np.rec.array(
        [(1, 101)],
        dtype=[("InvalidField", "int32"), ("StimNo", "int32")],
    )
    with pytest.raises(DH5Error, match="Invalid trialmap dtype"):
        add_trialmap_to_file(mock_h5_file, invalid_trialmap)


def test_add_trialmap_to_file_replace(mock_h5_file, valid_trialmap):
    first_trialmap = np.rec.array(
        [(1, 101, 0, 1000000000, 2000000000)], dtype=TRIALMAP_DATASET_DTYPE
    )
    add_trialmap_to_file(mock_h5_file, first_trialmap)
    assert np.array_equal(get_trialmap_from_file(mock_h5_file), first_trialmap)
    with pytest.raises(DH5Error):
        add_trialmap_to_file(mock_h5_file, valid_trialmap, replace=False)
    add_trialmap_to_file(mock_h5_file, valid_trialmap, replace=True)

    assert np.array_equal(get_trialmap_from_file(mock_h5_file), valid_trialmap)


def test_add_trialmap_to_file_no_replace(mock_h5_file, valid_trialmap):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    with pytest.raises(DH5Error):
        add_trialmap_to_file(mock_h5_file, valid_trialmap, replace=False)


def test_get_trialmap_from_file(mock_h5_file, valid_trialmap):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    retrieved_trialmap = get_trialmap_from_file(mock_h5_file)
    assert np.array_equal(retrieved_trialmap, valid_trialmap)


def test_get_trialmap_from_file_no_dataset(mock_h5_file):
    trialmap = get_trialmap_from_file(mock_h5_file)
    assert trialmap is None


def test_validate_trialmap(mock_h5_file, valid_trialmap, caplog):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    validate_trialmap(mock_h5_file)
    assert "TRIALMAP dataset not found" not in caplog.text


def test_validate_trialmap_no_dataset(mock_h5_file, caplog):
    validate_trialmap(mock_h5_file)
    assert "TRIALMAP dataset not found" in caplog.text


def test_validate_trialmap_dataset(mock_h5_file, valid_trialmap):
    add_trialmap_to_file(mock_h5_file, valid_trialmap)
    dataset = mock_h5_file[TRIALMAP_DATASET_NAME]
    validate_trialmap_dataset(dataset)


def test_validate_trialmap_dataset_invalid(mock_h5_file: h5py.File):
    invalid_dtype = [("InvalidField", "int32")]
    invalid_trialmap = mock_h5_file.create_dataset(
        TRIALMAP_DATASET_NAME, (1,), dtype=invalid_dtype
    )
    with pytest.raises(DH5Error):
        validate_trialmap_dataset(invalid_trialmap)


# tests for Trialmap class


def test_trialmap_initialization(valid_trialmap):
    trialmap = Trialmap(valid_trialmap)
    assert isinstance(trialmap, Trialmap)
    assert trialmap.recarray.dtype == TRIALMAP_DATASET_DTYPE
    assert len(trialmap.recarray) == len(valid_trialmap)
    assert len(trialmap) == len(valid_trialmap)


def test_trialmap_invalid_dtype():
    invalid_trialmap = np.rec.array(
        [(1, 101)],
        dtype=[("InvalidField", "int32"), ("StimNo", "int32")],
    )
    with pytest.raises(DH5Error, match="Invalid trialmap dtype"):
        Trialmap(invalid_trialmap)


def test_trialmap_properties(valid_trialmap):
    trialmap = Trialmap(valid_trialmap)
    assert np.array_equal(trialmap.recarray.TrialNo, valid_trialmap.TrialNo)
    assert np.array_equal(trialmap.recarray.StimNo, valid_trialmap.StimNo)
    assert np.array_equal(trialmap.recarray.Outcome, valid_trialmap.Outcome)
    assert np.array_equal(trialmap.recarray.StartTime, valid_trialmap.StartTime)
    assert np.array_equal(trialmap.recarray.EndTime, valid_trialmap.EndTime)

    # Check float conversion
    assert np.allclose(
        trialmap.start_time_float_seconds,
        valid_trialmap.StartTime.astype(np.float64) / 1e9,
    )
    assert np.allclose(
        trialmap.end_time_float_seconds, valid_trialmap.EndTime.astype(np.float64) / 1e9
    )

    # check properties
    assert np.array_equal(trialmap.trial_numbers, valid_trialmap.TrialNo)
    assert np.array_equal(trialmap.trial_type_numbers, valid_trialmap.StimNo)
    assert np.array_equal(trialmap.trial_outcomes_integer, valid_trialmap.Outcome)
    assert all(
        outcome in trialmap.trial_outcomes_as_enum for outcome in valid_trialmap.Outcome
    )


# tests for DEFAULT_OUTCOME_CODES and add_write_trialmap_operation


def test_default_outcome_codes_keys():
    """DEFAULT_OUTCOME_CODES uses BrainBox-compatible names."""
    assert set(DEFAULT_OUTCOME_CODES.keys()) == {
        "SUCCESS",
        "EARLY",
        "LATE",
        "EYE_ERROR",
    }


def test_default_outcome_codes_values_match_trial_outcome_enum():
    """DEFAULT_OUTCOME_CODES values are taken from the TrialOutcome enum."""
    assert DEFAULT_OUTCOME_CODES["SUCCESS"] == TrialOutcome.Hit
    assert DEFAULT_OUTCOME_CODES["EARLY"] == TrialOutcome.Early
    assert DEFAULT_OUTCOME_CODES["LATE"] == TrialOutcome.Late
    assert DEFAULT_OUTCOME_CODES["EYE_ERROR"] == TrialOutcome.EyeErr


# tests for add_write_trialmap_operation


EXAMPLE_OUTCOME_CODES = {
    "SUCCESS": 1,
    "EARLY": 5,
    "LATE": 6,
    "EYE_ERROR": 7,
}


def test_add_write_trialmap_operation_creates_group(mock_h5_file):
    """The operation group is created under Operations with the correct name."""
    add_write_trialmap_operation(mock_h5_file, EXAMPLE_OUTCOME_CODES)
    assert "Operations" in mock_h5_file
    ops = mock_h5_file["Operations"]
    keys = list(ops.keys())
    assert len(keys) == 1
    assert WRITE_TRIALMAP_OPERATION_NAME in keys[0]


def test_add_write_trialmap_operation_default_outcome_codes(mock_h5_file):
    """When outcome_codes is None, DEFAULT_OUTCOME_CODES (BrainBox names → TrialOutcome values) are used."""
    add_write_trialmap_operation(mock_h5_file)
    ops = mock_h5_file["Operations"]
    op_group = ops[list(ops.keys())[0]]
    assert op_group.attrs["SUCCESS"] == np.float64(TrialOutcome.Hit)
    assert op_group.attrs["EARLY"] == np.float64(TrialOutcome.Early)
    assert op_group.attrs["LATE"] == np.float64(TrialOutcome.Late)
    assert op_group.attrs["EYE_ERROR"] == np.float64(TrialOutcome.EyeErr)
    # No extra keys beyond the four defaults
    outcome_attr_keys = {"SUCCESS", "EARLY", "LATE", "EYE_ERROR"}
    assert outcome_attr_keys == outcome_attr_keys & set(op_group.attrs.keys())


def test_add_write_trialmap_operation_outcome_codes_are_float64(mock_h5_file):
    """Each outcome code attribute must be stored as float64 (BrainBox requirement)."""
    add_write_trialmap_operation(mock_h5_file, EXAMPLE_OUTCOME_CODES)
    ops = mock_h5_file["Operations"]
    op_group = ops[list(ops.keys())[0]]
    for name, code in EXAMPLE_OUTCOME_CODES.items():
        assert name in op_group.attrs, f"Missing attribute '{name}'"
        value = op_group.attrs[name]
        assert isinstance(value, np.float64), (
            f"Attribute '{name}' should be float64, got {type(value)}"
        )
        assert value == np.float64(code)


def test_add_write_trialmap_operation_default_tool(mock_h5_file):
    """The Tool attribute defaults to the canonical BrainBox tool string."""
    add_write_trialmap_operation(mock_h5_file, EXAMPLE_OUTCOME_CODES)
    ops = mock_h5_file["Operations"]
    op_group = ops[list(ops.keys())[0]]
    tool = op_group.attrs["Tool"]
    # stored as bytes in HDF5
    if isinstance(tool, bytes):
        tool = tool.decode()
    assert tool == WRITE_TRIALMAP_TOOL_NAME


def test_add_write_trialmap_operation_optional_tdr_file_and_tmfkt(mock_h5_file):
    """TDR_file and tmFkt optional attributes are written when supplied."""
    tdr = "/data/subject/session.tdr"
    tmfkt = "trialmap_Subject(fid)"
    add_write_trialmap_operation(
        mock_h5_file,
        EXAMPLE_OUTCOME_CODES,
        tdr_file=tdr,
        tmfkt=tmfkt,
    )
    ops = mock_h5_file["Operations"]
    op_group = ops[list(ops.keys())[0]]

    tdr_attr = op_group.attrs["TDR_file"]
    if isinstance(tdr_attr, bytes):
        tdr_attr = tdr_attr.decode()
    assert tdr_attr == tdr

    tmfkt_attr = op_group.attrs["tmFkt"]
    if isinstance(tmfkt_attr, bytes):
        tmfkt_attr = tmfkt_attr.decode()
    assert tmfkt_attr == tmfkt


def test_add_write_trialmap_operation_optional_attrs_absent_by_default(mock_h5_file):
    """TDR_file and tmFkt are not written when not supplied."""
    add_write_trialmap_operation(mock_h5_file, EXAMPLE_OUTCOME_CODES)
    ops = mock_h5_file["Operations"]
    op_group = ops[list(ops.keys())[0]]
    assert "TDR_file" not in op_group.attrs
    assert "tmFkt" not in op_group.attrs


def test_add_write_trialmap_operation_date(mock_h5_file):
    """The Date attribute reflects the supplied datetime."""
    fixed_date = datetime.datetime(2010, 1, 28, 13, 38, 23)
    add_write_trialmap_operation(mock_h5_file, EXAMPLE_OUTCOME_CODES, date=fixed_date)
    ops = mock_h5_file["Operations"]
    op_group = ops[list(ops.keys())[0]]
    date_arr = op_group.attrs["Date"]
    assert date_arr["Year"] == 2010
    assert date_arr["Month"] == 1
    assert date_arr["Day"] == 28
    assert date_arr["Hour"] == 13
    assert date_arr["Minute"] == 38
    assert date_arr["Second"] == 23


def test_add_write_trialmap_operation_writes_reference_outcome_codes_as_float64(tmp_path):
    """Write representative trialmap outcome codes and verify they are stored as float64 attrs."""
    reference_codes = {
        "SUCCESS": 0,
        "EARLY": 1,
        "LATE": 2,
        "EYE_ERROR": 3,
        "UNDETERMINED": 4,
        "REJ_SUCCESS": 100,
        "REJ_EARLY": 101,
        "REJ_LATE": 102,
        "REJ_EYE_ERROR": 103,
        "REJ_UNDETERMINED": 104,
        "TRIAL_END": -11,
    }
    file_path = tmp_path / "out.h5"
    with h5py.File(file_path, "w") as f:
        add_write_trialmap_operation(
            f,
            reference_codes,
            operator_name="Orlando",
            date=datetime.datetime(2010, 1, 28, 13, 38, 23),
            tdr_file="/cifs/venusData/AGS1/Versace/V_V5_066//Versace 2010_01_28 12_14_53.tdr",
            tmfkt="trialmap_Versace(fid)",
        )

    with h5py.File(file_path, "r") as f:
        ops = f["Operations"]
        op_group = ops[list(ops.keys())[0]]
        for name, code in reference_codes.items():
            value = op_group.attrs[name]
            assert isinstance(value, np.float64), f"{name} must be float64"
            assert value == np.float64(code), f"{name}: expected {code}, got {value}"
