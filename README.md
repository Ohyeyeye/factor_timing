# Factor Timing Strategy

This project implements a factor timing strategy that dynamically allocates weights to Fama-French factors based on macroeconomic and market conditions.

## Project Structure

```
factor_timing/
├── data/
│   └── data_loader.py      # Data loading and preprocessing
├── models/
│   └── regime_classifier.py # Regime classification models
├── optimization/
│   └── portfolio_optimizer.py # Portfolio optimization strategies
├── backtest/
│   └── backtester.py       # Backtesting framework
└── main.py                 # Main strategy implementation
```

## Features

- **Factor Universe**: Implements Fama-French 5-factor model
- **Regime Classification**: Multiple model options:
  - LSTM/GRU for sequential patterns
  - XGBoost for tabular data
  - HMM for probabilistic regime detection
- **Portfolio Optimization**:
  - Mean-variance optimization
  - Neural portfolio optimization
  - Regime-aware optimization
- **Backtesting**:
  - Monthly rebalancing
  - Performance metrics calculation
  - Visualization tools

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from factor_timing.main import FactorTimingStrategy

# Initialize strategy
strategy = FactorTimingStrategy(
    start_date='2010-01-01',
    end_date='2020-12-31',
    model_type='lstm',  # Options: 'lstm', 'xgboost', 'hmm'
    optimizer_type='mean_variance'  # Options: 'mean_variance', 'neural'
)

# Run strategy
results = strategy.run_strategy()

# Plot results
strategy.plot_results()
```

## Data Requirements

The strategy requires:
1. Fama-French 5-factor data
2. Macroeconomic data (GDP, CPI, interest rates, etc.)
3. Market data (S&P 500, VIX, etc.)

## Performance Metrics

The backtester calculates:
- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Tracking error (vs benchmark)

## Contributing

Feel free to submit issues and enhancement requests. 