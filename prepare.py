"""
Data preparation and backtest utilities for autoquant.

This file is FIXED - do not modify. It contains:
- Data download and caching
- Backtest execution engine
- Evaluation metrics

Agent only interacts via run_backtest() function.
"""

import os
from typing import Any, Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Fixed Constants (do not modify)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoquant")
DATA_PATH = os.path.join(CACHE_DIR, "sp100_prices.parquet")
SPY_DATA_PATH = os.path.join(CACHE_DIR, "spy_calendar.parquet")
MAPPING_PATH = os.path.join(CACHE_DIR, "ticker_mapping.json")

# Time periods
TRAIN_START = "2000-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2024-12-31"

# Trading parameters
GROSS_EXPOSURE_LIMIT = 3.0  # 300%
TRANSACTION_COST = 0.001  # 0.1%
RISK_FREE_RATE = 0.025  # 2.5% annual
INITIAL_CAPITAL = 1_000_000
MAX_FORWARD_FILL_DAYS = 5

# 60 highly liquid S&P 100 constituents with pre-2000 IPOs
TICKERS = [
    # Technology (12)
    "AAPL",
    "MSFT",
    "IBM",
    "INTC",
    "CSCO",
    "ORCL",
    "HPQ",
    "TXN",
    "ADP",
    "FISV",
    "KLAC",
    "AMAT",
    # Financials (10)
    "JPM",
    "BAC",
    "WFC",
    "C",
    "GS",
    "MS",
    "AXP",
    "BLK",
    "SPGI",
    "BK",
    # Healthcare (8)
    "JNJ",
    "PFE",
    "UNH",
    "ABT",
    "TMO",
    "MDT",
    "BMY",
    "LLY",
    # Consumer Discretionary (8)
    "WMT",
    "HD",
    "DIS",
    "MCD",
    "NKE",
    "LOW",
    "TGT",
    "BKNG",
    # Consumer Staples (6)
    "PG",
    "KO",
    "PEP",
    "COST",
    "PM",
    # "WBA", delisted
    # Industrials (6)
    "GE",
    "HON",
    "UNP",
    "UPS",
    "RTX",
    "LMT",
    # Energy (4)
    "XOM",
    "CVX",
    "COP",
    "SLB",
    # Materials (3)
    "LIN",
    "SHW",
    "NEM",
    # Communication (3)
    "VZ",
    "T",
    "CMCSA",
]

# Create anonymized ticker mapping: AAPL -> STK001, MSFT -> STK002, etc.
_ANONYMIZED_TICKERS = [f"STK{i + 1:03d}" for i in range(len(TICKERS))]
TICKER_TO_ANON = dict(zip(TICKERS, _ANONYMIZED_TICKERS))
ANON_TO_TICKER = dict(zip(_ANONYMIZED_TICKERS, TICKERS))

# ---------------------------------------------------------------------------
# Private Functions
# ---------------------------------------------------------------------------


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _save_ticker_mapping():
    """Save ticker mapping for reference."""
    import json

    mapping = {"real_to_anon": TICKER_TO_ANON, "anon_to_real": ANON_TO_TICKER}
    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Ticker mapping saved to: {MAPPING_PATH}")


def _download_spy_calendar() -> pd.DataFrame:
    """Download SPY data to use as trading calendar reference."""
    print("Downloading SPY calendar...")
    spy = yf.download(
        "SPY", start=TRAIN_START, end=TEST_END, progress=False, auto_adjust=False
    )
    if spy.empty:
        raise RuntimeError("Failed to download SPY data")

    spy_df = spy.reset_index()
    spy_df.columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    spy_df = spy_df[["date"]].drop_duplicates().sort_values("date")

    _ensure_cache_dir()
    spy_df.to_parquet(SPY_DATA_PATH)
    return spy_df


def _load_spy_calendar() -> pd.DatetimeIndex:
    """Load SPY trading calendar."""
    if os.path.exists(SPY_DATA_PATH):
        spy_df = pd.read_parquet(SPY_DATA_PATH)
    else:
        spy_df = _download_spy_calendar()
    return pd.DatetimeIndex(spy_df["date"])


def _download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single stock."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    df = df.reset_index()
    df.columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = ticker
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def _download_all_data() -> pd.DataFrame:
    """Download data for all tickers and align to SPY calendar."""
    print(f"Downloading data for {len(TICKERS)} tickers...")
    print("(Tickers are anonymized: AAPL -> STK001, MSFT -> STK002, etc.)")
    _ensure_cache_dir()

    # Get SPY calendar
    spy_dates = _load_spy_calendar()

    all_data = []
    failed_tickers = []

    for i, (real_ticker, anon_ticker) in enumerate(TICKER_TO_ANON.items()):
        try:
            print(f"  [{i + 1}/{len(TICKERS)}] {anon_ticker}...", end="")
            df = _download_stock_data(real_ticker, TRAIN_START, TEST_END)
            # Replace real ticker with anonymized name
            df["ticker"] = anon_ticker
            all_data.append(df)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({e})")
            failed_tickers.append(real_ticker)

    if failed_tickers:
        print(f"\nFailed to download: {failed_tickers}")
        raise RuntimeError(
            f"Failed to download {len(failed_tickers)} tickers: {failed_tickers}"
        )

    print("\nAligning to SPY calendar...")
    combined = pd.concat(all_data, ignore_index=True)

    # Create complete date x ticker grid using SPY dates
    spy_dates_df = pd.DataFrame({"date": spy_dates})
    tickers_df = pd.DataFrame({"ticker": list(TICKER_TO_ANON.values())})

    full_grid = spy_dates_df.merge(tickers_df, how="cross")

    # Merge with actual data
    merged = full_grid.merge(combined, on=["date", "ticker"], how="left")

    # Set index for forward-fill
    merged = merged.set_index(["date", "ticker"]).sort_index()

    return merged


def _apply_forward_fill(
    prices: pd.DataFrame, max_gap: int = MAX_FORWARD_FILL_DAYS
) -> pd.DataFrame:
    """
    Apply forward-fill with a maximum gap limit using pandas-native operations.
    If price is missing for more than max_gap days, leave as NaN (stock delisted).
    """
    # Group by ticker and apply forward fill with limit
    # This is much faster than Python loops (C-optimized)
    prices_filled = prices.groupby(level="ticker").ffill(limit=max_gap)

    # Set volume to 0 for filled days (where original was NaN but now has value)
    was_filled = prices["close"].isna() & prices_filled["close"].notna()
    prices_filled.loc[was_filled, "volume"] = 0

    return prices_filled


def _load_or_download_data() -> pd.DataFrame:
    """Load cached data or download if not exists."""
    if os.path.exists(DATA_PATH):
        print(f"Loading cached data from {DATA_PATH}")
        return pd.read_parquet(DATA_PATH)

    print("Cache miss - downloading data (this may take a few minutes)...")
    data = _download_all_data()

    print("Applying forward-fill...")
    data = _apply_forward_fill(data)

    # Drop any stocks with too many missing values (delisted early)
    valid_tickers = []
    for ticker in data.index.get_level_values("ticker").unique():
        ticker_data = data.loc[data.index.get_level_values("ticker") == ticker]
        missing_pct = ticker_data["close"].isna().mean()
        if missing_pct < 0.1:  # Keep if < 10% missing
            valid_tickers.append(ticker)
        else:
            print(f"  Dropping {ticker} ({missing_pct:.1%} missing data)")

    data = data.loc[data.index.get_level_values("ticker").isin(valid_tickers)]

    print(f"Saving to {DATA_PATH}...")
    data.to_parquet(DATA_PATH)

    # Save ticker mapping for reference
    _save_ticker_mapping()

    print("Done!")

    return data


def _calculate_sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 30:
        return float("-inf")

    # Weekly risk-free rate
    daily_rf = RISK_FREE_RATE / 252

    excess_returns = returns - daily_rf
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0 or np.isnan(std_excess):
        return float("-inf")

    # Annualized Sharpe
    sharpe = mean_excess / std_excess * np.sqrt(252)
    return sharpe


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown as percentage."""
    if len(returns) < 2:
        return float("-inf")

    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Calculate running maximum
    running_max = cum_returns.expanding().max()

    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max

    # Return minimum (most negative) drawdown
    return drawdown.min() * 100  # Convert to percentage


def _calculate_win_rate(returns: pd.Series) -> float:
    """Calculate percentage of positive return days."""
    if len(returns) == 0:
        return 0.0

    return (returns > 0).mean() * 100  # Convert to percentage


def run_backtest(
    setup: Callable[[pd.DataFrame], Any],
    calculate_portfolios: Callable[[pd.DataFrame, Any], pd.Series],
) -> tuple[str, dict]:
    """
    Run full backtest with separate setup and evaluation phases.

    Two-phase architecture:
    1. Setup phase: setup(train_data) is called once with training data
    2. Evaluation phase: calculate_portfolios(hist_data, model) is called daily
       with the prepared model

    Args:
        setup: Function that takes training DataFrame, returns any object
        calculate_portfolios: Function that takes historical DataFrame and model,
                             returns portfolio weights as Series

    Returns:
        tuple[str, dict]: (status, stats)
            status: "success", "setup_error", or "backtest_error"
            stats: Dictionary with train and test statistics
    """
    # Load data
    prices = _load_or_download_data()

    # Get trading calendar (SPY dates)
    spy_dates = prices.index.get_level_values("date").unique().sort_values()

    # Split train/test
    train_dates = spy_dates[spy_dates <= TRAIN_END]
    test_dates = spy_dates[(spy_dates >= TEST_START) & (spy_dates <= TEST_END)]
    all_dates = spy_dates[(spy_dates >= TRAIN_START) & (spy_dates <= TEST_END)]

    # Get list of valid tickers
    tickers = prices.index.get_level_values("ticker").unique().tolist()

    # === SETUP PHASE ===
    # Get training data (up to and including TRAIN_END)
    train_data = prices.loc[prices.index.get_level_values("date") <= TRAIN_END]

    try:
        model = setup(train_data.copy())
    except Exception as e:
        print(f"Error in setup: {e}")
        return "setup_error", {}

    # === EVALUATION PHASE ===
    # Initialize portfolio
    capital = INITIAL_CAPITAL
    positions = pd.Series(0.0, index=tickers)  # Dollar positions
    portfolio_value_history = []

    # Run backtest on all dates (train + test)
    for i, date in enumerate(all_dates):
        # Get prices up to and including current date
        hist_prices = prices.loc[prices.index.get_level_values("date") <= date]

        # Current prices (previous close)
        current_prices = prices.loc[prices.index.get_level_values("date") == date]
        current_prices = current_prices.reset_index("date", drop=True)["close"]
        current_prices = current_prices.reindex(tickers)

        # Skip if no valid prices
        if current_prices.isna().all():
            continue

        # Get target weights from portfolio calculation
        try:
            target_weights = calculate_portfolios(hist_prices.copy(), model)
            if target_weights is None or len(target_weights) == 0:
                target_weights = pd.Series(0.0, index=tickers)
        except Exception as e:
            print(f"Error in calculate_portfolios on {date}: {e}")
            return "backtest_error", {}

        # Ensure all tickers have weights
        target_weights = target_weights.reindex(tickers).fillna(0.0)

        # Normalize gross exposure to limit
        gross_exposure = target_weights.abs().sum()
        if gross_exposure > GROSS_EXPOSURE_LIMIT:
            target_weights = target_weights * (GROSS_EXPOSURE_LIMIT / gross_exposure)

        # Calculate cash weight (implicit)
        cash_weight = 1.0 - target_weights.abs().sum()

        # Target dollar positions
        target_positions = target_weights * capital

        # Calculate trades
        trades = target_positions - positions

        # Transaction costs on absolute trade size
        trade_value = trades.abs().sum()
        transaction_cost = trade_value * TRANSACTION_COST

        # Update positions
        positions = target_positions

        # Calculate cash
        cash = capital - positions.sum() - transaction_cost

        # Cash interest (applied daily)
        cash_interest = cash * (RISK_FREE_RATE / 252)
        cash += cash_interest

        # Calculate new portfolio value using next day's prices
        if i < len(all_dates) - 1:
            next_date = all_dates[i + 1]
            next_prices = prices.loc[prices.index.get_level_values("date") == next_date]
            next_prices = next_prices.reset_index("date", drop=True)["close"]
            next_prices = next_prices.reindex(tickers)

            # Calculate returns on positions
            shares = positions / current_prices.replace(0, np.nan)
            position_values = shares * next_prices
            position_values = position_values.fillna(0)

            # New total value
            capital = position_values.sum() + cash

        portfolio_value_history.append(
            {"date": date, "value": capital, "is_test": date in test_dates}
        )

    # Calculate returns
    portfolio_df = pd.DataFrame(portfolio_value_history).set_index("date")
    portfolio_df["return"] = portfolio_df["value"].pct_change()

    # Extract train and test period returns
    test_returns = portfolio_df.loc[portfolio_df["is_test"], "return"].dropna()
    train_returns = portfolio_df.loc[~portfolio_df["is_test"], "return"].dropna()

    # Calculate statistics for both periods
    stats = {
        "train": {
            "sharpe": _calculate_sharpe(train_returns)
            if len(train_returns) > 0
            else float("-inf"),
            "mean": train_returns.mean() * 100
            if len(train_returns) > 0
            else 0.0,  # Convert to %
            "std": train_returns.std() * 100 if len(train_returns) > 0 else 0.0,
            "skew": train_returns.skew() if len(train_returns) > 0 else 0.0,
            "max_drawdown": _calculate_max_drawdown(train_returns)
            if len(train_returns) > 0
            else float("-inf"),
            "win_rate": _calculate_win_rate(train_returns)
            if len(train_returns) > 0
            else 0.0,
        },
        "test": {
            "sharpe": _calculate_sharpe(test_returns)
            if len(test_returns) > 0
            else float("-inf"),
            "mean": test_returns.mean() * 100 if len(test_returns) > 0 else 0.0,
            "std": test_returns.std() * 100 if len(test_returns) > 0 else 0.0,
            "skew": test_returns.skew() if len(test_returns) > 0 else 0.0,
            "max_drawdown": _calculate_max_drawdown(test_returns)
            if len(test_returns) > 0
            else float("-inf"),
            "win_rate": _calculate_win_rate(test_returns)
            if len(test_returns) > 0
            else 0.0,
        },
    }

    return "success", stats


def run_training(train_fn: Callable[[pd.DataFrame], Any]) -> tuple[str, Any]:
    """
    Run training-only mode.

    Loads training data and calls the provided training function.
    This is for experimentation - no backtest is run.

    Args:
        train_fn: Function that takes training DataFrame, returns any object
                  for inspection (parameters, statistics, fitted model, etc.)

    Returns:
        tuple[str, Any]: (status, result)
            status: "success" or "error"
            result: Object returned by train_fn, or None on error
    """
    # Load data
    prices = _load_or_download_data()

    # Extract training period only
    train_data = prices.loc[prices.index.get_level_values("date") <= TRAIN_END]

    try:
        result = train_fn(train_data.copy())
        return "success", result
    except Exception as e:
        print(f"Error in training: {e}")
        return "error", None


# ---------------------------------------------------------------------------
# Main (for standalone data download)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("AutoQuant Data Preparation")
    print("=" * 50)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Tickers: {len(TICKERS)} (anonymized as STK001-STK{len(TICKERS):03d})")
    print(f"Date range: {TRAIN_START} to {TEST_END}")
    print()

    # Just download and cache the data
    data = _load_or_download_data()

    print()
    print("Data preparation complete!")
    print(f"Data saved to: {DATA_PATH}")
    print(f"Ticker mapping saved to: {MAPPING_PATH}")
    print(f"Shape: {data.shape}")
    print(f"Tickers: {len(data.index.get_level_values('ticker').unique())}")
    print(
        f"Date range: {data.index.get_level_values('date').min()} to {data.index.get_level_values('date').max()}"
    )
