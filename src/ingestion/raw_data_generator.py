"""
Generate synthetic trader activity data for the ML platform.

This module simulates trader personas, trade events, PnL, risk behavior, and
account evolution used by downstream feature engineering and modeling.
"""

import random
import uuid

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from src.ingestion.personas import TRADER_PERSONAS

class SyntheticTradeGenerator:
    """Generate synthetic trade histories from configurable trader personas."""

    def __init__(self, num_traders=200, start_date="2025-01-01", num_days=60):
        """
        Initialize generator settings and deterministic trader identifiers.

        Args:
            num_traders: Number of synthetic traders to generate.
            start_date: First business day for generated trades.
            num_days: Number of business days to simulate.
        """

        self.num_traders = num_traders
        self.start_date = datetime.strptime(start_date,"%Y-%m-%d")

        self.num_days = num_days

        self.trader_ids = [f"TRADER_{i}" 
                           for i in range(1, num_traders + 1)]
        
        self.minimum_balance_threshold = 10000
   

    def assign_persona(self):
        """
        Randomly assign one configured behavior persona.

        Returns:
            Persona key from TRADER_PERSONAS.
        """
        return random.choice(list(TRADER_PERSONAS.keys()))
    

    def generate_trader_profiles(self):
        """
        Generate starting account profiles for all synthetic traders.

        Returns:
            DataFrame with trader ID, persona, account size, and balance.
        """

        trader_profiles = []

        for trader_id in self.trader_ids:
            persona = self.assign_persona()
            account_size = self.generate_account_size(persona)
            trader_profiles.append({
                'trader_id': trader_id,
                'persona': persona,
                "account_size": account_size,
                "current_balance": account_size
            })
        return pd.DataFrame(trader_profiles)
    

    def generate_trade_count(self, persona, trading_day):
        """
        Generate the number of trades for a persona on a trading day.

        Args:
            persona: Trader persona key.
            trading_day: Business date being simulated.

        Returns:
            Positive integer trade count.
        """

        persona_config = TRADER_PERSONAS[persona]
        avg_trades = persona_config["avg_trades_per_day"]

        if self.is_expiry_day(trading_day):

            # Expiry days amplify activity for short-term and speculative personas.
            if persona == "expiry_day_gambler":
                avg_trades *= 2

            elif persona == "scalper_option_buyer":
                avg_trades *= 1.5

        return max(1, 
                   int(np.random.normal(
                       loc = avg_trades,
                       scale = avg_trades * 0.3
                   )))
    

    def generate_trade_timestamp(self, trading_day):
        """
        Generate an intraday timestamp during Indian market hours.

        Args:
            trading_day: Business date for the trade.

        Returns:
            Datetime representing the simulated trade time.
        """

        base_date = trading_day

        market_open_hour = 9
        market_open_minute = 15

        random_minutes = random.randint(0, 375)

        trade_timestamp = base_date.replace(
            hour=market_open_hour,
            minute=market_open_minute
        ) + timedelta(minutes=random_minutes)

        return trade_timestamp
    

    def generate_pnl(self, persona, leverage):
        """
        Generate trade PnL from persona-specific win/loss distributions.

        Args:
            persona: Trader persona key.
            leverage: Leverage multiplier applied to the trade.

        Returns:
            Rounded profit or loss value.
        """

        persona_config = TRADER_PERSONAS[persona]

        win_probability = persona_config["win_probability"]

        is_win = random.random() < win_probability

        # Each persona has a distinct payoff distribution to create separable behavior.
        if persona == "scalper_option_buyer":
            if is_win:
                pnl = np.random.normal(loc=800, scale=500)
            else:
                pnl = np.random.normal(loc=-800, scale=600)

        elif persona == "disciplined_option_seller":
            if is_win:
                pnl = np.random.normal(loc=800, scale=400)
            else:
                pnl = np.random.normal(loc=-1000, scale=600)

        elif persona == "spread_trader":
            if is_win:
                pnl = np.random.normal(loc=900, scale=400)
            else:
                pnl = np.random.normal(loc=-700, scale=300)

        elif persona == "expiry_day_gambler":
            if is_win:
                pnl = np.random.normal(loc=2500, scale=1500)
            else:
                pnl = np.random.normal(loc=-1800, scale=1000)

        else:
            pnl = np.random.normal(loc=100, scale=1000)

        pnl = pnl * leverage

        return round(pnl, 2)
    

    def generate_trade_event(self,trader_id,persona,trading_day,account_size, current_balance):
        """
        Generate one complete synthetic trade event.

        Args:
            trader_id: Synthetic trader identifier.
            persona: Trader persona key.
            trading_day: Business date for the trade.
            account_size: Starting account size for risk calculations.
            current_balance: Account balance before the trade.

        Returns:
            Dictionary containing trade-level fields used by the platform.
        """

        persona_config = TRADER_PERSONAS[persona]

        trade_id = str(uuid.uuid4())

        strategy = random.choice(persona_config["preferred_strategies"])

        leverage = round(random.uniform(*persona_config["leverage_range"]),2)

        risk_percentage = random.uniform(*persona_config["risk_per_trade_range"])

        if self.is_expiry_day(trading_day):

            # Increase risk on expiry days for personas expected to trade more aggressively.
            if persona == "expiry_day_gambler":
                risk_percentage *= 1.5

            elif persona == "scalper_option_buyer":
                risk_percentage *= 1.2

        capital_risked = round(account_size * risk_percentage, 2)

        holding_minutes = random.randint(*persona_config["holding_minutes_range"])

        stop_loss_used = int(random.random() < persona_config["stop_loss_discipline"])

        pnl = self.generate_pnl(persona,leverage)

        balance_before_trade = current_balance

        balance_after_trade = round(current_balance + pnl, 2)

        trade_timestamp = self.generate_trade_timestamp(trading_day)

        trade_outcome = "WIN" if pnl > 0 else "LOSS"

        overnight_position = int(random.random()<persona_config["overnight_holding_probability"])

        return {
            "trade_id": trade_id,
            "trader_id": trader_id,
            "persona": persona,
            "account_size": account_size,
            "balance_before_trade": balance_before_trade,
            "balance_after_trade": balance_after_trade,
            "trade_timestamp": trade_timestamp,
            "strategy_type": strategy,
            "leverage_used": leverage,
            "holding_minutes": holding_minutes,
            "stop_loss_used": stop_loss_used,
            "pnl": pnl,
            "trade_outcome": trade_outcome,
            "overnight_position": overnight_position,
            "capital_risked": capital_risked,
            "risk_percentage": round(risk_percentage, 4)
            }
        

    def generate_trading_days(self):
        """
        Generate business days for the simulation period.

        Returns:
            DatetimeIndex of trading days excluding weekends.
        """

        trading_days = pd.bdate_range(start=self.start_date, periods=self.num_days)
        return trading_days
    
    def is_expiry_day(self, trading_day):
        """
        Determine whether the given date is treated as an expiry day.

        Args:
            trading_day: Date to evaluate.

        Returns:
            True when the day is Thursday, otherwise False.
        """

        return trading_day.weekday() == 3
    

    def generate_trade_events(self):
        """
        Generate the full synthetic trade-event dataset.

        Returns:
            DataFrame of trade events across all traders and trading days.
        """
        all_trades = []
        trader_profiles = (self.generate_trader_profiles())

        for _, trader in trader_profiles.iterrows():
            trader_id = trader["trader_id"]
            persona = trader["persona"]
            account_size = trader["account_size"]
            current_balance = trader["current_balance"]
            
            trading_days = (self.generate_trading_days())
            for trading_day in trading_days:

                # Stop simulating traders once their balance falls below a viability threshold.
                if current_balance <= self.minimum_balance_threshold:
                    break

                num_trades = self.generate_trade_count(persona, trading_day)

                for _ in range(num_trades):
                    trade_event = self.generate_trade_event(
                        trader_id, 
                        persona, 
                        trading_day, 
                        account_size, 
                        current_balance)

                    all_trades.append(trade_event)
                    current_balance = trade_event["balance_after_trade"]

        return pd.DataFrame(all_trades)
    
    def generate_account_size(self, persona):
        """
        Generate persona-specific starting account size.

        Args:
            persona: Trader persona key.

        Returns:
            Integer account size sampled from persona-specific ranges.
        """

        if persona == "scalper_option_buyer":
            return random.randint(50000, 300000)

        elif persona == "disciplined_option_seller":
            return random.randint(500000, 3000000)

        elif persona == "spread_trader":
            return random.randint(200000, 1000000)

        elif persona == "expiry_day_gambler":
            return random.randint(20000, 150000)
        
        else:
            return random.randint(50000, 500000)
    



if __name__ == "__main__":

    generator = SyntheticTradeGenerator()

    trades_df = generator.generate_trade_events()

    print("\nDataset Shape:", trades_df.shape)

    print("\nTotal Trades Generated:")
    print(len(trades_df))

    print("\nSample Trades:")
    print(trades_df.head())

    trades_df.to_csv(
        "data/raw/raw_data_trades.csv",
        index=False
    )

    print("\nDataset saved to data/raw/raw_data_trades.csv")
