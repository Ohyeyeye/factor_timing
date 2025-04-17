import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self,
                 factor_returns: pd.DataFrame,
                 initial_capital: float = 1000000.0,
                 rebalance_freq: str = 'M'):
        self.factor_returns = factor_returns
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.portfolio_value = pd.Series(dtype=float)
        self.weights_history = pd.DataFrame()
        
    def run_backtest(self,
                    weights: pd.DataFrame,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict:
        """Run backtest with given weights"""
        if start_date is None:
            start_date = weights.index[0]
        if end_date is None:
            end_date = weights.index[-1]
            
        # Initialize portfolio
        portfolio_value = self.initial_capital
        portfolio_values = []
        dates = []
        
        # Get rebalance dates
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_freq)
        
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]
            
            # Get current weights
            current_weights = weights.loc[current_date]
            
            # Get returns for the period
            period_returns = self.factor_returns.loc[current_date:next_date]
            
            # Calculate portfolio returns
            portfolio_returns = (period_returns * current_weights).sum(axis=1)
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_returns).prod()
            
            # Store results
            portfolio_values.append(portfolio_value)
            dates.append(next_date)
            
        # Create results DataFrame
        self.portfolio_value = pd.Series(portfolio_values, index=dates)
        self.weights_history = weights
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        returns = self.portfolio_value.pct_change().dropna()
        
        # Calculate metrics
        total_return = (self.portfolio_value.iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        # Calculate maximum drawdown
        cummax = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - cummax) / cummax
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def plot_results(self, benchmark: Optional[pd.Series] = None):
        """Plot backtest results"""
        plt.figure(figsize=(12, 6))
        
        # Plot portfolio value
        plt.plot(self.portfolio_value.index, self.portfolio_value / self.initial_capital,
                label='Strategy', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark is not None:
            plt.plot(benchmark.index, benchmark / benchmark.iloc[0],
                    label='Benchmark', linewidth=2, alpha=0.7)
            
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot weights over time
        if not self.weights_history.empty:
            plt.figure(figsize=(12, 6))
            self.weights_history.plot(kind='area', stacked=True)
            plt.title('Portfolio Weights Over Time')
            plt.xlabel('Date')
            plt.ylabel('Weight')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.show()
    
    def compare_with_benchmark(self, benchmark: pd.Series) -> Dict:
        """Compare strategy with benchmark"""
        strategy_metrics = self.calculate_performance_metrics()
        
        # Calculate benchmark metrics
        benchmark_returns = benchmark.pct_change().dropna()
        benchmark_total_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
        benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
        benchmark_annualized_vol = benchmark_returns.std() * np.sqrt(252)
        benchmark_sharpe = benchmark_annualized_return / benchmark_annualized_vol if benchmark_annualized_vol != 0 else 0
        
        # Calculate tracking error
        tracking_error = (benchmark_returns - self.portfolio_value.pct_change().dropna()).std() * np.sqrt(252)
        
        return {
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': {
                'total_return': benchmark_total_return,
                'annualized_return': benchmark_annualized_return,
                'annualized_volatility': benchmark_annualized_vol,
                'sharpe_ratio': benchmark_sharpe
            },
            'tracking_error': tracking_error
        } 