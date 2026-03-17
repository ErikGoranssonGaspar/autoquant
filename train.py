"""
train.py - Training-only experimentation mode for autoquant.

Run with: uv run python train.py

This file allows you to experiment with training data, fit models,
and tune hyperparameters WITHOUT running the full backtest.

Workflow:
1. Implement your train() function below with your training logic
2. Run `uv run python train.py` to test it
3. Iterate quickly - no backtest overhead
4. Once satisfied, port your logic to trade.py's setup() and calculate_portfolios()
5. Run full backtest with `uv run python trade.py`

Output format:
  TRAIN_SUCCESS | time: 45.2s
  Return value: {'param1': 0.5, 'model_weights': array([...])}

  TRAIN_ERROR: <message> | time: 12.3s

  TRAIN_TIMEOUT | time: 300.0s

Best practices:
- Use this to test feature engineering ideas quickly
- Print diagnostics to understand what's happening
- Compare multiple parameter sets before committing to backtest
- Much faster than trade.py - use it liberally during exploration
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

from prepare import run_training


# Timeout handler
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Training exceeded 5 minute limit")


def train(train_data: pd.DataFrame) -> Any:
    """
    Training/experimentation function - edit this!

    Called ONCE with full training period data. Use this to:
    - Test feature engineering ideas
    - Fit statistical models
    - Tune hyperparameters
    - Explore data patterns
    - Validate assumptions

    This is NOT the backtest - it's a sandbox for experimentation.
    Whatever you return will be printed for inspection.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame with MultiIndex (date, ticker), columns: open, high, low, close, volume
        Contains ONLY training period data (2000-2020)

    Returns
    -------
    Any
        Return any object - dict, model, parameters, statistics, etc.
        The return value will be displayed after training completes.
        Use this to inspect what your code discovered.
    """

    # === YOUR CODE HERE ===
    # Experiment with training data
    # Print diagnostics to understand patterns
    # Test different approaches quickly

    return None

if __name__ == "__main__":
    start_time = time.time()

    # Set 5 minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 300 seconds = 5 minutes

    try:
        # Run training mode
        status, result = run_training(train)

        # Cancel timeout
        signal.alarm(0)

        # Calculate elapsed time
        elapsed = time.time() - start_time

        # Print formatted output based on status
        if status == "success":
            print(f"TRAIN_SUCCESS | time: {elapsed:.1f}s")
            print(f"Return value: {repr(result)[:500]}")  # Truncate if too long
        elif status == "error":
            print(f"TRAIN_ERROR: Training failed | time: {elapsed:.1f}s")
            sys.exit(1)
        else:
            print(f"ERROR: Unknown status {status} | time: {elapsed:.1f}s")
            sys.exit(1)

    except TimeoutException:
        elapsed = time.time() - start_time
        print(f"TRAIN_TIMEOUT | time: {elapsed:.1f}s")
        sys.exit(1)

    except Exception as e:
        # Cancel timeout
        signal.alarm(0)

        elapsed = time.time() - start_time
        print(f"TRAIN_ERROR: {e} | time: {elapsed:.1f}s")

        # Print full traceback
        traceback.print_exc()
        sys.exit(1)
