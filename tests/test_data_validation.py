"""Tests for raw trade dataset validation rules and reporting."""

import pandas as pd
import pytest

from src.validation.validate_dataset import validate_dataset


REQUIRED_TRADE_COLUMNS = {
    "trade_id",
    "trader_id",
    "persona",
    "account_size",
    "balance_before_trade",
    "balance_after_trade",
    "trade_timestamp",
    "strategy_type",
    "leverage_used",
    "holding_minutes",
    "stop_loss_used",
    "pnl",
    "trade_outcome",
    "overnight_position",
    "capital_risked",
    "risk_percentage",
}


def assert_valid_trade_dataset(df: pd.DataFrame) -> None:
    """Assert the raw trade dataset satisfies the expected test contract."""
    missing = REQUIRED_TRADE_COLUMNS.difference(df.columns)
    assert not missing
    assert df["trader_id"].notna().all()
    assert not df["trade_id"].duplicated().any()
    assert df["leverage_used"].between(0.01, 20.0).all()
    assert pd.to_datetime(df["trade_timestamp"], errors="coerce").notna().all()


def test_valid_trade_data_passes_schema_checks(sample_raw_trade_data):
    """Verify valid sample trade data passes schema and quality checks."""
    # Purpose: validate the happy path contract required by feature engineering.
    assert_valid_trade_dataset(sample_raw_trade_data)


def test_missing_required_column_is_detected(sample_raw_trade_data):
    """Verify missing required columns are detected."""
    # Purpose: ensure required source columns are not silently accepted when absent.
    invalid = sample_raw_trade_data.drop(columns=["trade_id"])

    with pytest.raises(AssertionError):
        assert_valid_trade_dataset(invalid)


def test_null_trader_id_is_detected(sample_raw_trade_data):
    """Verify null trader identifiers are detected."""
    # Purpose: verify primary entity identifiers must be present.
    invalid = sample_raw_trade_data.copy()
    invalid.loc[0, "trader_id"] = None

    with pytest.raises(AssertionError):
        assert_valid_trade_dataset(invalid)


def test_duplicate_trade_id_is_detected(sample_raw_trade_data):
    """Verify duplicate trade IDs are detected."""
    # Purpose: protect the trade_id primary key from duplicate source events.
    invalid = sample_raw_trade_data.copy()
    invalid.loc[1, "trade_id"] = invalid.loc[0, "trade_id"]

    with pytest.raises(AssertionError):
        assert_valid_trade_dataset(invalid)


def test_out_of_range_leverage_is_detected(sample_raw_trade_data):
    """Verify invalid leverage values are detected."""
    # Purpose: catch impossible leverage values before modeling.
    invalid = sample_raw_trade_data.copy()
    invalid.loc[0, "leverage_used"] = 0.0

    with pytest.raises(AssertionError):
        assert_valid_trade_dataset(invalid)


def test_unparseable_timestamp_is_detected(sample_raw_trade_data):
    """Verify unparseable timestamps are detected."""
    # Purpose: verify timestamps can be parsed for time-based features.
    invalid = sample_raw_trade_data.copy()
    invalid.loc[0, "trade_timestamp"] = "not-a-date"

    with pytest.raises(AssertionError):
        assert_valid_trade_dataset(invalid)


def test_validate_dataset_reports_duplicate_and_balance_errors(sample_raw_trade_data, capsys):
    """Verify the production validation report surfaces data quality errors."""
    # Purpose: exercise the production validation report on an invalid dataset.
    invalid = sample_raw_trade_data.copy()
    invalid.loc[1, "trade_id"] = invalid.loc[0, "trade_id"]
    invalid.loc[2, "balance_after_trade"] = 1.0

    validate_dataset(invalid)

    report = capsys.readouterr().out
    assert "Duplicate Trade IDs: 1" in report
    assert "Balance Errors: 1" in report
