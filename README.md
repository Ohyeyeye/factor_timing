# Factor Timing Strategy

A factor timing strategy implementation that uses machine learning to identify market regimes and optimize portfolio weights accordingly.

## Project Structure

```
factor_timing/
├── predictor/           # Regime prediction models
├── optimizer/           # Portfolio optimization methods
├── backtest/           # Backtesting framework
├── data/               # Data loading and processing
├── main.py            # Main strategy implementation
├── select_strategy.py # Strategy selection and comparison
└── requirements.txt   # Project dependencies
```

## Current System Configuration

The current implementation uses:
- **LSTM Regime Predictor**: For market regime prediction
- **Neural Attention Optimizer**: For portfolio weight optimization

### Model Management

The system comes with pretrained models that are aligned with the following time periods:
- Training: 2000-2019
- Testing: 2020-2024

To test different time frameworks:
1. Delete the pretrained models from the `models/` directory
2. Update the date ranges in `main.py`
3. Run the training process again

## Features

- **Factor Universe**: Implements Fama-French 5-factor model
- **Regime Prediction**: LSTM-based regime detection
- **Portfolio Optimization**: Neural attention-based optimization
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

To run the factor timing strategy:

```bash
python main.py
```

This will:
1. Load and prepare the required data
2. Train the LSTM Regime Predictor and Neural Attention Optimizer (if pretrained models don't exist)
3. Run the strategy on the test period
4. Calculate and display performance metrics
5. Generate performance visualizations

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
