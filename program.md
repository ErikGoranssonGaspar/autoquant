# autoquant

Autonomous quantitative trading research.

This is an experiment to have an LLM agent develop quantitative trading approaches through iterative experimentation.

## Quick Start

**Typical workflow:**

1. **Verify setup**: Fresh git branch (e.g., `autoquant/mar15`), data cached at `~/.cache/autoquant/`
2. **Experiment with train.py**: Test ideas quickly (~10-30s per run, no git commits needed)
3. **Backtest with trade.py**: Evaluate on test data (~2-3 min per run, log to results.tsv)
4. **Iterate**: Keep commits that improve test Sharpe, reset others

```bash
# Quick experimentation - use this liberally
uv run python train.py

# Full backtest - only when ready to evaluate
uv run python trade.py
```

## The Goal

**Maximize the Sharpe ratio on the test period ONLY**.

The Sharpe ratio measures risk-adjusted returns - higher is better. A positive Sharpe means your approach beats the risk-free rate. Typical range: -1 to +2.

Both train and test statistics are displayed, but **only test Sharpe matters for optimization**. Train metrics help diagnose overfitting (when train Sharpe is much higher than test Sharpe).

## Architecture

### Two-Phase Backtest System

**Setup Phase** (`setup()` in trade.py): Called ONCE with training data
- Prepare any state needed for the evaluation phase  
- Return an object that will be passed to calculate_portfolios()
- Setup time counts toward 5-minute timeout
- **Print diagnostics here** - log parameter values, feature statistics, validation results

**Evaluation Phase** (`calculate_portfolios()` in trade.py): Called daily during backtest
- Receives historical data + object from setup
- Returns portfolio weights for that day
- Object persists across all daily calls

This separation prevents look-ahead bias.

### Key Files

| File | Purpose | You Edit? |
|------|---------|-----------|
| **trade.py** | Full backtest - implements setup() and calculate_portfolios() | YES |
| **train.py** | Training-only experimentation - implements train() function | YES |
| **prepare.py** | Data loading and backtest engine | NO - forbidden |
| **results.tsv** | Experiment log (NOT committed to git) | Append only |

## Recommended Workflow

### Phase 1: Experimentation (train.py)

Use `train.py` for rapid iteration before committing to full backtests.

1. **Implement idea** in the `train()` function
2. **Run training**: `uv run python train.py` (~10-30 seconds)
3. **Print diagnostics** - see what patterns emerge
4. **Iterate freely** - no git commits needed, just edit and re-run
5. **Compare approaches** - try multiple ideas quickly

**When to use train.py:**
- Testing feature engineering ideas
- Fitting statistical models
- Hyperparameter search
- Debugging logic
- Exploring data patterns

### Phase 2: Evaluation (trade.py)

Once you have a promising approach, evaluate it properly.

**LOOP FOREVER:**

1. **Port to trade.py** - Move logic to `setup()` and/or `calculate_portfolios()`
2. **Check git state** - Note current branch and commit
3. **Read results.tsv** - Find best Sharpe achieved (from `status=keep` rows)
4. **git commit** your changes
5. **Run backtest**: `uv run python trade.py`
6. **Parse output** - Extract test Sharpe and runtime
7. **Log to results.tsv** (tab-separated)
8. **Evaluate**:
   - If Sharpe **improved**: Keep commit (`status=keep`)
   - If Sharpe **worse/equal**: `git reset --hard HEAD~1` to undo
   - If **crash**: Attempt fix, or reset if unfixable

### Key Principle

Only keep commits that improve test Sharpe. Failed experiments are still logged to results.tsv for reference, but code is reset via `git reset --hard HEAD~1`.

**Never go back to main** unless starting completely fresh.

## Files Reference

### trade.py

Implements the trading strategy. You edit two functions:

- `setup(train_data)` - Called once with training data, returns model object
- `calculate_portfolios(prices_df, model)` - Called daily, returns portfolio weights

**Constraints:**
- Only edit these two functions (plus helper functions you add)
- Available libraries: numpy, pandas, scipy, statsmodels, ta-lib
- 5-minute total timeout (setup + evaluation)

### train.py

Experimentation sandbox. Implement the `train()` function:

- Called once with training data only
- Returns any object for inspection
- ~10-30 second runtime (vs 2-3 minutes for trade.py)
- No git commits needed
- Not logged to results.tsv

**Best practice**: Use this liberally during exploration. Print everything. Compare approaches. Once you find something promising, port minimal code to trade.py.

### results.tsv

Local experiment log (NOT committed to git). Columns (tab-separated):

```
timestamp	commit	sharpe	time_seconds	status	description
```

**Status values:**
- `baseline` - Initial run
- `keep` - Improved Sharpe, keep this commit
- `discard` - Equal or worse Sharpe, discard
- `crash` - Runtime error
- `timeout` - Exceeded 5 minute limit

**Example:**
```
timestamp	commit	sharpe	time_seconds	status	description
2024-03-15 10:30:00	a1b2c3d	0.389784	97.1	baseline	equal weight baseline
2024-03-15 11:00:00	b2c3d4e	0.452123	98.5	keep	added size factor
```

## Interface Reference

### Function Signatures

**`setup(train_data: pd.DataFrame) -> Any`**

Called once at the start with training data.

**Parameters:**
- `train_data`: DataFrame with MultiIndex (date, ticker), columns: open, high, low, close, volume

**Returns:**
- Any object - dict, parameters, or None
- This object is passed to `calculate_portfolios()` on every day

**`calculate_portfolios(prices_df: pd.DataFrame, model: Any) -> pd.Series`**

Called daily during backtest.

**Parameters:**
- `prices_df`: DataFrame with all historical data up to current date
- `model`: Object returned by `setup()`

**Returns:**
- `pd.Series`: Target portfolio weights indexed by ticker (STK001, STK002, etc.)
  - Positive = long position
  - Negative = short position
  - Zero = no position
- Cash is implicit (residual of 1.0 - sum(abs(weights)))

### Data Format

Both functions receive pandas DataFrames with:
- **MultiIndex**: `(date, ticker)` where date is a DatetimeIndex
- **Columns**: open, high, low, close, volume

**Frequency**: Daily trading days (not intraday/minute data).
**Time Range**: Over 20 years of historical data (2000-2024).
**Tickers**: Approximately 60 anonymized stocks (STK001, STK002, ..., STK060).

**Note**: This is long-term daily data spanning multiple market cycles. Strategies should focus on multi-day to multi-week holding periods. High-frequency or intraday strategies are not supported by this data granularity.

**Important**: You don't know which dates are in train vs test. The Sharpe ratio is computed ONLY on the test period, but you receive all historical data up to each day.

### Code Examples

**Get close prices:**
```python
closes = prices_df['close'].unstack('ticker')  # Wide format: dates x tickers
latest = closes.iloc[-1]  # Latest prices for all tickers
```

**Select specific ticker:**
```python
ticker_data = prices_df.loc[prices_df.index.get_level_values('ticker') == 'STK001']
```

### Market Parameters

- **Transaction cost**: 0.1% on absolute position changes
- **Cash yield**: 2.5% annually (paid daily)
- **Gross exposure limit**: 300% (silently enforced)
- **Rebalancing**: Daily at previous close

## Output Format

### trade.py (Full Backtest)

Success:
```
SHARPE    train: 0.923456  test: 0.823456
MEAN      train: 0.052%    test: 0.041%
STD       train: 0.802%    test: 0.753%
SKEW      train: -0.234    test: -0.152
DRAWDOWN  train: -15.2%    test: -22.1%
WIN_RATE  train: 52.3%     test: 51.8%
TIME      145.2s
```

**IMPORTANT**: Only the **test** Sharpe ratio matters for optimization.

Errors:
```
TIMEOUT | time: 300.0s
SETUP_ERROR: <message> | time: 12.3s
BACKTEST_ERROR: <message> | time: 12.3s
ERROR: <message> | time: 12.3s
```

### train.py (Training Only)

Success:
```
TRAIN_SUCCESS | time: 25.3s
Return value: {'lookback': 20, 'mean_return': 0.0004}
```

Errors:
```
TRAIN_TIMEOUT | time: 300.0s
TRAIN_ERROR: <message> | time: 12.3s
```

## Best Practices

### Workflow Strategy

**Start with train.py:**
- Use it liberally during exploration - it's your sandbox
- Print everything to understand what's happening
- Compare multiple parameter sets quickly
- Don't over-optimize here - it's just for discovery

**Then verify with trade.py:**
- Port minimal code from train.py
- Always verify promising ideas with full backtest
- Watch for overfitting (train Sharpe >> test Sharpe)
- Test Sharpe is the only metric that matters

### Training vs. Execution

The `setup()` function often **overfits** to training data:

- Use `setup()` to validate broad hypotheses quickly
- Automated parameter optimization in `setup()` frequently finds parameters that don't generalize
- Document discovered parameters, then **hardcode** them in `calculate_portfolios()` for efficiency and stability
- Manual parameter search (e.g., testing thresholds 0.5%, 0.75%, 1%) often outperforms automated training

### Statistical Significance

Not all improvements are real:

- Changes <0.01 Sharpe are likely noise — don't chase marginal gains
- If optimizing beyond 2 significant digits (e.g., 0.685% vs 0.69%), you're overfitting
- Test robustness: if a parameter change of ±10% destroys performance, it's not robust
- Improvements should be **consistent** across multiple runs if real

### Time Budget Awareness

The 5-minute timeout is real:

- Complex feature engineering in `setup()` easily hits timeout
- **Pre-calculate expensive features once in `setup()`**, not daily in `calculate_portfolios()`
- Test runtime on a subset before full backtest if possible
- Remember: `setup()` time counts toward the 5-minute limit

### When to Pivot

If you're stuck, follow these decision rules:

- **10-experiment rule**: If 10 attempts show no improvement, try a completely different approach
- **Opposite test**: If approach X doesn't work, try -X (e.g., short instead of long, low instead of high)
- **Simplification test**: Remove your last added feature — if Sharpe stays same or improves, keep the simpler version
- **Radical change**: If 20+ attempts with no progress, abandon the current line of research entirely

### Getting Unstuck

If your Sharpe ratio plateaus:

**1. Analyze results.tsv:**
- Which changes improved Sharpe? Which didn't?
- What were the magnitudes? (0.01 vs 0.10 improvements)
- Are there patterns in what works?

**2. Re-read your code:**
- Check `calculate_portfolios()` docstring for correct usage
- Verify no future leakage (using data you shouldn't have)
- Handle edge cases (empty weights, all NaN, etc.)

**3. Experiment radically:**
- Try the opposite of what you've been doing
- Change timescales completely
- Mix two approaches that both had partial success
- Research online for new quantitative trading ideas

### Logging Discipline

- **Log EVERY experiment immediately** after `git commit` and before evaluation
- Include both successes AND failures in `results.tsv`
- Use results.tsv to identify patterns, not just track your best score
- Look for clusters: what approaches consistently work or fail?
- Failed experiments teach you what NOT to do

### Simplicity Criterion

All else being equal, simpler is better:

- A 0.01 Sharpe improvement that adds 50 lines of messy code? Probably not worth it.
- A 0.01 Sharpe improvement from deleting code? Definitely keep.
- An improvement of ~0 but much simpler code? Keep.

The same goes for speed and computational efficiency.

## Crash Recovery

If your code crashes:

1. **Read the traceback carefully** — most crashes are simple bugs
2. **Common issues**:
   - `KeyError`: Accessing a ticker that doesn't exist
   - `ValueError`: Shape mismatch in operations
   - `ZeroDivisionError`: Dividing by zero
   - `TypeError`: Wrong data types
3. **Fix the bug** — usually just a few lines need changing
4. **Re-run** — don't give up until you've tried to fix it

**DO NOT** mark as `crash` and abandon without attempting a fix. Most crashes are simple errors.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep or away and expects you to work *indefinitely* until manually stopped.

You are autonomous. If you run out of ideas:
- Re-read this document
- Look at your results.tsv for patterns
- Try combining previous attempts
- Try more radical changes
- Research online for new approaches

The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~3 minutes, you can run approximately 20/hour, for about 160 experiments over an 8-hour sleep.

## Summary

- **Quick experiment**: `uv run python train.py` (~10-30s, no commits)
- **Full backtest**: `uv run python trade.py` (~2-3 min, log to results.tsv)
- **Goal**: Maximize test Sharpe only
- **Keep**: Only commits that improve test Sharpe
- **Reset**: `git reset --hard HEAD~1` when Sharpe doesn't improve
- **Persist**: Never stop until manually interrupted

Good luck! May your Sharpe ratios be high and your drawdowns low.
