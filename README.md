# AutoQuant
A minimal backtesting environment for AI agents to develop optimal trading strategies by editing just two files — in a loop, forever. Think of it as a numerical optimizer, not over subset of $\mathbb{R}^n$, but on the space of all trading strategies expressible in code. Based on [Andrej Karpathy's *autoresearcher*](https://github.com/karpathy/autoresearch). 

## The Loop
The user runs `prepare.py`, which downloads a stock price data set and sets up a simple backtesting engine; this script is not modified by the agent. Its interface is described in `program.md`, which also contains rules and best practices for the agent. It is instructed to 

1. Edit and run `train.py`, which gives it access to a training data set. Once it has found a promising trading approach it will
2. Implement this strategy in `trade.py` and commit it to git. The agent will then run this script, which runs the backtesting engine and returns the test set Sharpe ratio, along with some other metrics.
3. The agent logs this result in `results.tsv` (and creates it if it doesn't yet exist).
4. If the new strategy improves on the previous best it is kept; the agent keeps iterating. Otherwise it resets to the previous commit and tries again.

This keeps going forever, until the user interrupts.

## Quick Start
To download stock data and setup the backtesting environment, first install the project dependencies using the `uv` package manager by running
```bash
uv sync
```
Then run the preparation script:
```bash
uv run prepare.py
```
Finally, make sure everything is working by running a back-test using the default baseline buy-and-hold strategy:
```bash
uv run train.py
```
To launch the autoresearcher simply point your preferred agentic coding harness (Claude Code / Codex / OpenCode / ...) to this repo with a prompt like
```
Hi! You are a quantitative trading researcher tasked with finding a trading strategy which maximizes the Sharpe ratio on historical stock prices. Have a look at program.md and get started. Keep working forever and never stop to ask for input. 
```

## Implementation Details
The setup script `prepare.py` downloads stock prices for a hardcoded list of ca. 60 S&P100 companies from Yahoo finance. The training set consist of data from 2000 to 2020 and the test set on years 2021–2024. 

To avoid the agent getting stuck, both `train.py` and `trade.py` will timeout if they run for more than five minutes. This implemented in the scripts themselves, which means that the agent *could* decide to alter the timeout limit, but I have not had this happen. 

 Ideally the agent keeps running forever, although sometimes it insists on stopping, contrary to its instructions. This seems to depend on which model and agentic harness you use and the exact wording of the instructions in `program.md`. Agents sometimes also forget to record their progress in `results.tsv`; they then need to be reminded.

 I recommend enforcing the editing rules by your agentic harness' permissions system. Full edit permissions should be granted only for `train.py`, `trade.py` and `results.tsv`. Make sure that the agent cannot edit `prepare.py` and `program.md`. I have included an example config for OpenCode in `opencode.json`.

For the sake of simplicity, I have not allowed the agent to install its own dependencies. This is enforced by the OpenCode configuration. The agent has access to `numpy`, `pandas`, `ta-lib`, `statsmodels`, `scipy` and `scikit-learn`.  
