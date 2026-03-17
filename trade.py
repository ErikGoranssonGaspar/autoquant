"""
trade.py - Trading strategy file for autoquant.

This is the ONLY file the agent edits.
Run with: uv run python trade.py

Output format:
  sharpe: <value> | time: <seconds>s

Examples:
  sharpe: 0.823456 | time: 145.2s   (success)
  TIMEOUT | time: 300.0s            (exceeded 5 minute limit)
  SETUP_ERROR: <message> | time: 12.3s    (error during setup phase)
  BACKTEST_ERROR: <message> | time: 12.3s (error during backtest phase)
"""

import signal
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import talib as ta
import sklearn

from prepare import run_backtest


# Timeout handler
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Backtest exceeded 5 minute limit")


def setup(train_data: pd.DataFrame) -> Any:
    """
    Prepare any state needed for portfolio calculation.

    Called once with training data. Return any object you need later,
    or None if you don't need to store anything.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame with MultiIndex (date, ticker), columns: open, high, low, close, volume

    Returns
    -------
    Any
        Object passed to calculate_portfolios() on each day.
        Can be None, dict, or any custom object.
    """
    # Equal weight baseline - no state needed
    return None


def calculate_portfolios(prices_df: pd.DataFrame, model: Any) -> pd.Series:
    """
    Calculate target portfolio weights for next trading day.

    Called daily during backtest with all historical data up to current date.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with MultiIndex (date, ticker), columns: open, high, low, close, volume

        Data access:
        >>> closes = prices_df['close'].unstack('ticker')
        >>> latest = closes.iloc[-1]

    model : Any
        Object returned by setup().

    Returns
    -------
    pd.Series
        Target portfolio weights indexed by ticker.
        Positive = long, Negative = short, Zero = no position.
        Gross exposure normalized to 300% if exceeded.
    """
    # Get all available tickers from the data
    closes = prices_df["close"].unstack("ticker")

    # Equal weight across all stocks
    # Each stock gets 1/N of the portfolio where N = number of stocks
    weights = pd.Series(1.0, index=closes.columns)

    # Normalize so weights sum to 1.0 (100% invested, 0% cash)
    weights = weights / weights.sum()

    return weights


if __name__ == "__main__":
    start_time = time.time()

    # Set 5 minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 300 seconds = 5 minutes

    try:
        # Run the backtest with both functions
        status, stats = run_backtest(setup, calculate_portfolios)

        # Cancel timeout
        signal.alarm(0)

        # Calculate elapsed time
        elapsed = time.time() - start_time

        # Print formatted output based on status
        if status == "success":
            print("\nBacktest Results")
            print("=" * 60)
            print(f"{'Metric':<20} {'Train':>18} {'Test':>18}")
            print("-" * 60)
            print(
                f"{'Sharpe Ratio':<20} {stats['train']['sharpe']:>18.6f} {stats['test']['sharpe']:>18.6f}"
            )
            print(
                f"{'Mean Return (%)':<20} {stats['train']['mean']:>17.4f}% {stats['test']['mean']:>17.4f}%"
            )
            print(
                f"{'Std Dev (%)':<20} {stats['train']['std']:>17.4f}% {stats['test']['std']:>17.4f}%"
            )
            print(
                f"{'Skewness':<20} {stats['train']['skew']:>18.4f} {stats['test']['skew']:>18.4f}"
            )
            print(
                f"{'Max Drawdown (%)':<20} {stats['train']['max_drawdown']:>17.2f}% {stats['test']['max_drawdown']:>17.2f}%"
            )
            print(
                f"{'Win Rate (%)':<20} {stats['train']['win_rate']:>17.2f}% {stats['test']['win_rate']:>17.2f}%"
            )
            print("-" * 60)
            print(f"Elapsed time: {elapsed:.1f}s")
        elif status == "setup_error":
            print(f"SETUP_ERROR: Setup failed | time: {elapsed:.1f}s")
            sys.exit(1)
        elif status == "backtest_error":
            print(
                f"BACKTEST_ERROR: Portfolio calculation failed | time: {elapsed:.1f}s"
            )
            sys.exit(1)
        else:
            print(f"ERROR: Unknown status {status} | time: {elapsed:.1f}s")
            sys.exit(1)

    except TimeoutException:
        elapsed = time.time() - start_time
        print(f"TIMEOUT | time: {elapsed:.1f}s")
        sys.exit(1)

    except Exception as e:
        # Cancel timeout
        signal.alarm(0)

        elapsed = time.time() - start_time
        print(f"ERROR: {e} | time: {elapsed:.1f}s")

        # Print full traceback so agent can debug errors
        traceback.print_exc()
        sys.exit(1)
