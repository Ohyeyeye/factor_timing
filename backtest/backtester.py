import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from data.data_loader import DataLoader

class Backtester:
    def __init__(self, returns: pd.DataFrame, rebalance_freq: str = 'ME'):
        """
        Initialize backtester
        
        Args:
            returns (pd.DataFrame): Factor returns (SMB, HML, RMW, CMA are actual returns, Mkt-RF is excess return)
            rebalance_freq (str): Rebalancing frequency (e.g., 'ME' for month-end)
        """
        self.returns = returns
        self.rebalance_freq = rebalance_freq
        self.portfolio_values = None
        self.portfolio_returns = None
        self.weights = None
        self.logger = logging.getLogger(__name__)
        
        # Load risk-free rate for Sharpe ratio calculation
        try:
            # Get the date range from the returns data
            start_date = returns.index.min().strftime('%Y-%m-%d')
            end_date = returns.index.max().strftime('%Y-%m-%d')
            
            # Load Fama-French factors using DataLoader
            data_loader = DataLoader()
            ff_data = data_loader.load_fama_french_factors(start_date, end_date)
            
            # Convert index to datetime and get RF rate
            ff_data.index = pd.to_datetime(ff_data.index)
            self.rf_rate = ff_data['RF'].apply(pd.to_numeric, errors='coerce') / 100  # Convert from percentage to decimal
            self.logger.info("Loaded risk-free rate from FF5 data")
            self.logger.info(f"RF rate range: {self.rf_rate.min():.4%} to {self.rf_rate.max():.4%}")
        except Exception as e:
            self.logger.warning(f"Could not load risk-free rate from FF5 data: {e}. Will use default value.")
            self.rf_rate = None
        
    def run_backtest(self, returns: pd.DataFrame, weights: pd.DataFrame) -> Dict:
        """Run backtest and return performance metrics"""
        self.logger.info("Starting backtest...")
        
        # Store returns and weights
        self.returns = returns
        self.weights = weights
        
        # Calculate portfolio returns
        self.logger.info("Calculating portfolio returns...")
        portfolio_returns = self._calculate_portfolio_returns(returns, weights)
        
        # Calculate performance metrics
        self.logger.info("Calculating performance metrics...")
        metrics = self.calculate_performance_metrics(portfolio_returns)
        
        # Store results
        self.results = metrics
        
        self.logger.info("Backtest completed")
        return metrics
        
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        total_days = (returns.index[-1] - returns.index[0]).days
        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (365 / total_days) - 1
        
    def _calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)  # Assuming daily returns
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
            
        # Calculate excess returns using risk-free rate
        excess_returns = returns - self.rf_rate
        
        # Calculate annualized Sharpe ratio
        annualized_sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        return float(annualized_sharpe)
        
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = portfolio_values.expanding().max()
        drawdowns = portfolio_values / rolling_max - 1
        return drawdowns.min()
        
    def plot_results(self, benchmarks: Optional[Dict[str, pd.Series]] = None):
        """Plot backtest results with multiple benchmarks"""
        self.logger.info("Starting to plot results...")
        if self.portfolio_values is None:
            raise ValueError("Run backtest first before plotting")
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Portfolio Value Over Time
        ax1.plot(self.portfolio_values.index, self.portfolio_values.values / 1000, 
                label='Strategy', linewidth=2)
        
        if benchmarks is not None:
            self.logger.info("Adding benchmarks to plot...")
            for name, returns in benchmarks.items():
                # Calculate cumulative value starting from $1000
                benchmark_value = 1000 * (1 + returns).cumprod()
                ax1.plot(benchmark_value.index, benchmark_value.values / 1000, 
                        label=name, linestyle='--', alpha=0.7)
            
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value (thousands)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Monthly Returns Comparison
        # Resample returns to monthly frequency using ME (month end) instead of M
        monthly_returns = self.portfolio_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        if benchmarks is not None:
            # Plot monthly returns comparison
            width = 0.25
            x = np.arange(len(monthly_returns))
            
            # Strategy returns
            ax2.bar(x - width, monthly_returns * 100, width, 
                   label='Strategy', alpha=0.7)
            
            # Benchmark returns
            for i, (name, returns) in enumerate(benchmarks.items()):
                # Align benchmark returns with strategy returns
                aligned_returns = returns.reindex(self.portfolio_returns.index)
                # Resample to monthly frequency
                monthly_benchmark = aligned_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                # Ensure same length as strategy returns
                monthly_benchmark = monthly_benchmark.reindex(monthly_returns.index)
                ax2.bar(x + width * i, monthly_benchmark * 100, width, 
                       label=name, alpha=0.7)
        else:
            # Plot only strategy monthly returns
            ax2.bar(monthly_returns.index, monthly_returns * 100, label='Strategy')
            
        ax2.set_title('Monthly Returns')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()
        self.logger.info("Plot completed")
        
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate detailed performance metrics"""
        self.logger.info("Starting performance metrics calculation...")
        if self.portfolio_returns is None:
            raise ValueError("Run backtest first before calculating metrics")
            
        self.logger.info("Calculating metrics...")
        
        # Calculate total return
        total_return = (1 + returns).prod() - 1
        
        # Calculate turnover
        turnover = self.weights.diff().abs().sum(axis=1).mean() * 252  # Annualized turnover
        
        metrics = {
            'total_return': total_return,
            'annualized_return': self._calculate_annualized_return(returns),
            'annualized_volatility': self._calculate_annualized_volatility(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(self.portfolio_values),
            'turnover': turnover,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01)
        }
        
        self.logger.info("Performance metrics calculation completed")
        return metrics
        
    def compare_with_benchmark(self, benchmarks: Dict[str, pd.Series]) -> Dict:
        """Compare strategy performance with multiple benchmarks"""
        if self.portfolio_returns is None:
            raise ValueError("Run backtest first before comparing with benchmarks")
            
        comparison = {}
        
        for name, benchmark_returns in benchmarks.items():
            # Align benchmark with portfolio returns
            if isinstance(benchmark_returns, pd.Series):
                aligned_benchmark = benchmark_returns.reindex(self.portfolio_returns.index)
            else:
                # If benchmark is a float, create a Series with the same index as portfolio returns
                aligned_benchmark = pd.Series(benchmark_returns, index=self.portfolio_returns.index)
            
            # Convert both series to numpy arrays and ensure they're float type
            portfolio_array = self.portfolio_returns.values.astype(float)
            benchmark_array = aligned_benchmark.values.astype(float)
            
            # Calculate tracking error
            tracking_error = (self.portfolio_returns - aligned_benchmark).std() * np.sqrt(252)
            
            # Calculate information ratio
            active_returns = self.portfolio_returns - aligned_benchmark
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
            
            # Calculate beta
            covariance = np.cov(portfolio_array, benchmark_array)[0,1]
            variance = np.var(benchmark_array)
            beta = covariance / variance if variance > 0 else 0
            
            # Calculate alpha
            alpha = self.portfolio_returns.mean() - beta * aligned_benchmark.mean()
            
            comparison[name] = {
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'correlation': self.portfolio_returns.corr(aligned_benchmark)
            }
        
        return comparison

    def _calculate_portfolio_returns(self, returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from factor returns and weights"""
        self.logger.info("Calculating portfolio returns...")
        
        # Ensure returns and weights are aligned
        common_idx = returns.index.intersection(weights.index)
        returns = returns.loc[common_idx]
        weights = weights.loc[common_idx]
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Store portfolio returns and values
        self.portfolio_returns = portfolio_returns
        self.portfolio_values = (1 + portfolio_returns).cumprod() * 1000  # Starting with $1000
        
        self.logger.info(f"Calculated {len(portfolio_returns)} portfolio returns")
        return portfolio_returns 