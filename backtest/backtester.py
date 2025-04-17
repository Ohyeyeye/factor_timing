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
        self.portfolio_values = None
        self.portfolio_returns = None
        self.weights = None
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, weights: pd.DataFrame) -> Dict:
        """Run backtest on the test period"""
        self.logger.info("Starting backtest...")
        
        # Ensure dates are aligned
        self.logger.info("Aligning dates between weights and returns...")
        common_dates = weights.index.intersection(self.returns.index)
        weights = weights.loc[common_dates]
        returns = self.returns.loc[common_dates]
        self.logger.info(f"Aligned {len(common_dates)} common dates")
        
        # Validate returns
        self.logger.info("Validating returns...")
        if (returns > 1.0).any().any():
            self.logger.warning("Found returns > 100%, capping at 100%")
            returns = returns.clip(upper=1.0)
        self.logger.info("Returns validation completed")
        
        # Initialize portfolio value
        self.logger.info("Initializing portfolio...")
        portfolio_value = 1.0
        portfolio_values = []
        
        # Run backtest
        self.logger.info("Running backtest simulation...")
        for date in weights.index:
            current_weights = weights.loc[date]
            current_returns = returns.loc[date]
            
            # Validate weights
            if not np.isclose(current_weights.sum(), 1.0, rtol=1e-5):
                self.logger.warning(f"Weights do not sum to 1 on {date}, normalizing")
                current_weights = current_weights / current_weights.sum()
            
            # Calculate daily return
            daily_return = (current_weights * current_returns).sum()
            
            # Cap daily return at 100%
            daily_return = min(daily_return, 1.0)
            
            portfolio_value *= (1 + daily_return)
            portfolio_values.append(portfolio_value)
            
            # Log if return is unusually high
            if daily_return > 0.5:  # 50% daily return
                self.logger.warning(f"Unusually high return on {date}: {daily_return:.2%}")
        
        self.logger.info("Backtest simulation completed")
        
        # Store results as instance variables
        self.logger.info("Storing backtest results...")
        self.portfolio_values = pd.Series(portfolio_values, index=weights.index)
        self.portfolio_returns = pd.Series([(portfolio_values[i]/portfolio_values[i-1] - 1) 
                                          for i in range(1, len(portfolio_values))], 
                                          index=weights.index[1:])
        self.weights = weights
        
        # Create results dictionary
        results = {
            'portfolio_values': self.portfolio_values,
            'returns': self.portfolio_returns,
            'weights': self.weights
        }
        
        self.logger.info("Backtest completed successfully")
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
        self.logger.info("Starting to plot results...")
        if self.portfolio_values is None:
            raise ValueError("Run backtest first before plotting")
            
        self.logger.info("Creating plot...")
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_values.index, self.portfolio_values.values, label='Strategy')
        
        if benchmark is not None:
            self.logger.info("Adding benchmark to plot...")
            # Align benchmark with portfolio values
            aligned_benchmark = benchmark.reindex(self.portfolio_values.index)
            plt.plot(aligned_benchmark.index, aligned_benchmark.values, label='Benchmark')
            
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        self.logger.info("Plot completed")
        
    def calculate_performance_metrics(self) -> Dict:
        """Calculate detailed performance metrics"""
        self.logger.info("Starting performance metrics calculation...")
        if self.portfolio_returns is None:
            raise ValueError("Run backtest first before calculating metrics")
            
        self.logger.info("Calculating metrics...")
        metrics = {
            'annualized_return': self._calculate_annualized_return(self.portfolio_returns),
            'annualized_volatility': self._calculate_annualized_volatility(self.portfolio_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(self.portfolio_returns),
            'max_drawdown': self._calculate_max_drawdown(self.portfolio_values),
            'skewness': self.portfolio_returns.skew(),
            'kurtosis': self.portfolio_returns.kurtosis(),
            'var_95': self.portfolio_returns.quantile(0.05),
            'var_99': self.portfolio_returns.quantile(0.01)
        }
        
        self.logger.info("Performance metrics calculation completed")
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