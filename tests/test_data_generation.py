"""Tests for synthetic trader profile and trade-event generation."""

import random

import numpy as np
import pandas as pd

from src.ingestion.personas import TRADER_PERSONAS
from src.ingestion.raw_data_generator import SyntheticTradeGenerator


def test_generate_trader_profiles_returns_expected_schema(monkeypatch):
    """Verify generated trader profiles have the expected schema and values."""
    # Purpose: verify deterministic trader profile creation without touching persisted data.
    generator = SyntheticTradeGenerator(num_traders=3, start_date="2025-01-01", num_days=1)
    monkeypatch.setattr(generator, "assign_persona", lambda: "scalper_option_buyer")
    monkeypatch.setattr(generator, "generate_account_size", lambda persona: 75_000)

    profiles = generator.generate_trader_profiles()

    assert profiles.shape == (3, 4)
    assert profiles["trader_id"].tolist() == ["TRADER_1", "TRADER_2", "TRADER_3"]
    assert set(profiles["persona"]) == {"scalper_option_buyer"}
    assert profiles["current_balance"].eq(75_000).all()


def test_generate_trade_event_returns_valid_trade_fields(monkeypatch):
    """Verify a generated trade event contains required downstream fields."""
    # Purpose: validate generated trade events contain the contract expected downstream.
    generator = SyntheticTradeGenerator(num_traders=1, start_date="2025-01-02", num_days=1)
    monkeypatch.setattr(generator, "generate_pnl", lambda persona, leverage: 500.0)
    monkeypatch.setattr(generator, "generate_trade_timestamp", lambda day: pd.Timestamp(day).to_pydatetime())

    trade = generator.generate_trade_event(
        trader_id="TRADER_1",
        persona="disciplined_option_seller",
        trading_day=pd.Timestamp("2025-01-02"),
        account_size=100_000,
        current_balance=100_000,
    )

    assert set(
        [
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
        ]
    ).issubset(trade)
    assert trade["balance_after_trade"] == 100_500
    assert trade["trade_outcome"] == "WIN"
    assert trade["leverage_used"] > 0


def test_generate_trade_events_is_deterministic_with_seed():
    """Verify seeded generation is reproducible apart from UUID trade IDs."""
    # Purpose: keep seeded synthetic fields reproducible while allowing UUID trade IDs.
    random.seed(7)
    np.random.seed(7)
    first_generator = SyntheticTradeGenerator(num_traders=2, start_date="2025-01-01", num_days=2)
    first = first_generator.generate_trade_events()

    random.seed(7)
    np.random.seed(7)
    second_generator = SyntheticTradeGenerator(num_traders=2, start_date="2025-01-01", num_days=2)
    second = second_generator.generate_trade_events()

    pd.testing.assert_frame_equal(first.drop(columns=["trade_id"]), second.drop(columns=["trade_id"]))
    assert first["trade_id"].is_unique
    assert second["trade_id"].is_unique


def test_generate_trade_count_is_positive_for_all_personas():
    """Verify every persona generates at least one trade per active day."""
    # Purpose: ensure downstream aggregations never receive zero generated trades per day.
    generator = SyntheticTradeGenerator(num_traders=1, start_date="2025-01-01", num_days=1)

    counts = [
        generator.generate_trade_count(persona, pd.Timestamp("2025-01-01"))
        for persona in TRADER_PERSONAS
    ]

    assert all(count >= 1 for count in counts)


def test_generate_trading_days_excludes_weekends():
    """Verify generated trading days use business-day frequency."""
    # Purpose: verify generated market days use business-day frequency.
    generator = SyntheticTradeGenerator(num_traders=1, start_date="2025-01-01", num_days=5)

    trading_days = generator.generate_trading_days()

    assert len(trading_days) == 5
    assert all(day.weekday() < 5 for day in trading_days)
