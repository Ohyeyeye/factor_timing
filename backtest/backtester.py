import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import logging

class Backtester:
    def __init__(self, returns: pd.DataFrame, rebalance_freq: str = 'ME'):
        """
        Initialize backtester
        
        Args:
            returns (pd.DataFrame): Asset returns
            rebalance_freq (str): Rebalancing frequency (e.g., 'ME' for month-end)
        """
        self.returns = returns
        self.rebalance_freq = rebalance_freq
        self.portfolio_returns = None
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, weights: pd.DataFrame) -> Dict:
        """Run backtest on the test period"""
        self.logger.info("Starting backtest...")
        
        # Ensure dates are aligned
        common_dates = weights.index.intersection(self.returns.index)
        weights = weights.loc[common_dates]
        returns = self.returns.loc[common_dates]
        
        # Initialize portfolio value
        portfolio_value = 1.0
        portfolio_values = []
        
        # Run backtest
        for date in weights.index:
            current_weights = weights.loc[date]
            current_returns = returns.loc[date]
            
            # Calculate daily return
            daily_return = (current_weights * current_returns).sum()
            portfolio_value *= (1 + daily_return)
            portfolio_values.append(portfolio_value)
        
        # Create results dictionary
        results = {
            'portfolio_values': pd.Series(portfolio_values, index=weights.index),
            'returns': pd.Series([(portfolio_values[i]/portfolio_values[i-1] - 1) 
                                for i in range(1, len(portfolio_values))], 
                                index=weights.index[1:]),
            'weights': weights
        }
        
        self.logger.info("Backtest completed")
        return results
        
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        total_days = (returns.index[-1] - returns.index[0]).days
        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (365 / total_days) - 1
        
    def _calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)  # Assuming daily returns
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = portfolio_values.expanding().max()
        drawdowns = portfolio_values / rolling_max - 1
        return drawdowns.min()
        
    def plot_results(self, benchmark: Optional[pd.Series] = None):
        """Plot backtest results"""
        if self.portfolio_returns is None:
            raise ValueError("Run backtest first before plotting")
            
        plt.figure(figsize=(12, 6))
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values, label='Strategy')
        
        if benchmark is not None:
            # Align benchmark with portfolio returns
            aligned_benchmark = benchmark.reindex(self.portfolio_returns.index)
            cumulative_benchmark = (1 + aligned_benchmark).cumprod()
            plt.plot(cumulative_benchmark.index, cumulative_benchmark.values, label='Benchmark')
            
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def calculate_performance_metrics(self) -> Dict:
        """Calculate detailed performance metrics"""
        if self.portfolio_returns is None:
            raise ValueError("Run backtest first before calculating metrics")
            
        metrics = {
            'annualized_return': self._calculate_annualized_return(self.portfolio_returns),
            'annualized_volatility': self._calculate_annualized_volatility(self.portfolio_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(self.portfolio_returns),
            'max_drawdown': self._calculate_max_drawdown((1 + self.portfolio_returns).cumprod()),
            'skewness': self.portfolio_returns.skew(),
            'kurtosis': self.portfolio_returns.kurtosis(),
            'var_95': self.portfolio_returns.quantile(0.05),
            'var_99': self.portfolio_returns.quantile(0.01)
        }
        
        return metrics
        
    def compare_with_benchmark(self, benchmark: pd.Series) -> Dict:
        """Compare strategy performance with benchmark"""
        if self.portfolio_returns is None:
            raise ValueError("Run backtest first before comparing with benchmark")
            
        # Align benchmark with portfolio returns
        aligned_benchmark = benchmark.reindex(self.portfolio_returns.index)
        
        # Calculate tracking error
        tracking_error = (self.portfolio_returns - aligned_benchmark).std() * np.sqrt(252)
        
        # Calculate information ratio
        active_returns = self.portfolio_returns - aligned_benchmark
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
        
        # Calculate beta
        covariance = np.cov(self.portfolio_returns, aligned_benchmark)[0,1]
        variance = aligned_benchmark.var()
        beta = covariance / variance
        
        comparison = {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'correlation': self.portfolio_returns.corr(aligned_benchmark)
        }
        
        return comparison 