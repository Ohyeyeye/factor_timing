import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from data.data_loader import DataLoader

class Backtester:
    def __init__(self, returns: pd.DataFrame, rebalance_freq: str = 'M'):
        """
        Initialize backtester
        
        Args:
            returns (pd.DataFrame): Factor returns (SMB, HML, RMW, CMA are actual returns, Mkt-RF is excess return)
            rebalance_freq (str): Rebalancing frequency (e.g., 'M' for month-end)
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
        
    def run_backtest(self, weights: pd.DataFrame) -> Dict:
        """Run backtest on the test period"""
        self.logger.info("Starting backtest...")
        
        # Ensure dates are aligned
        self.logger.info("Aligning dates between weights and returns...")
        common_dates = weights.index.intersection(self.returns.index)
        weights = weights.loc[common_dates]
        returns = self.returns.loc[common_dates]
        self.logger.info(f"Aligned {len(common_dates)} common dates")
        
        # Initialize portfolio value
        self.logger.info("Initializing portfolio...")
        portfolio_value = 1000
        portfolio_values = []
        
        # Run backtest
        self.logger.info("Running backtest simulation...")
        for date in weights.index:
            current_weights = weights.loc[date]
            current_returns = returns.loc[date]
            
            # Validate weights
            weight_sum = current_weights.sum()
            if not np.isclose(weight_sum, 1.0, rtol=1e-5):
                self.logger.warning(f"Weights do not sum to 1 on {date}, normalizing")
                if weight_sum > 0:
                    current_weights = current_weights / weight_sum
                else:
                    self.logger.warning("All weights are zero, using equal weights")
                    current_weights = pd.Series(1/len(current_weights), index=current_weights.index)
            
            # Ensure weights are non-negative
            if (current_weights < 0).any():
                self.logger.warning(f"Negative weights found on {date}, clipping to 0")
                current_weights = current_weights.clip(lower=0)
                weight_sum = current_weights.sum()
                if weight_sum > 0:
                    current_weights = current_weights / weight_sum
                else:
                    self.logger.warning("All weights are zero after clipping, using equal weights")
                    current_weights = pd.Series(1/len(current_weights), index=current_weights.index)
            
            # Calculate daily return as weighted sum of factor returns
            daily_return = (current_weights * current_returns).sum()
            
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
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio using FF5 risk-free rate"""
        if self.rf_rate is not None:
            try:
                # Align RF rate with returns
                aligned_rf = self.rf_rate.reindex(returns.index)
                # For any missing values, use the previous available RF rate
                aligned_rf = aligned_rf.fillna(method='ffill')
                
                # Calculate excess returns using actual RF rate
                excess_returns = returns - aligned_rf
                
                # Log some diagnostic information
                self.logger.info(f"Returns mean: {returns.mean():.4%}")
                self.logger.info(f"RF rate mean: {aligned_rf.mean():.4%}")
                self.logger.info(f"Excess returns mean: {excess_returns.mean():.4%}")
                self.logger.info(f"Returns std: {returns.std():.4%}")
                
                # Check for valid data
                if excess_returns.std() == 0:
                    self.logger.warning("Zero standard deviation in excess returns")
                    return 0
                
                annualized_sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                self.logger.info(f"Calculated Sharpe ratio: {annualized_sharpe:.4f}")
                return annualized_sharpe
                
            except Exception as e:
                self.logger.error(f"Error calculating Sharpe ratio: {e}")
                return 0
        else:
            # Fallback to default RF rate if FF5 data not available
            self.logger.warning("Using default risk-free rate of 2%")
            excess_returns = returns - 0.02/252  # Daily risk-free rate
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
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
        # Resample returns to monthly frequency using M (month end) instead of M
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
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
                monthly_benchmark = aligned_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
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
        
    def compare_with_benchmark(self, benchmarks: Dict[str, pd.Series]) -> Dict:
        """Compare strategy performance with multiple benchmarks"""
        if self.portfolio_returns is None:
            raise ValueError("Run backtest first before comparing with benchmarks")
            
        comparison = {}
        
        for name, benchmark_returns in benchmarks.items():
            # Align benchmark with portfolio returns
            aligned_benchmark = benchmark_returns.reindex(self.portfolio_returns.index)
            
            # Calculate tracking error
            tracking_error = (self.portfolio_returns - aligned_benchmark).std() * np.sqrt(252)
            
            # Calculate information ratio
            active_returns = self.portfolio_returns - aligned_benchmark
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
            
            # Calculate beta
            covariance = np.cov(self.portfolio_returns, aligned_benchmark)[0,1]
            variance = aligned_benchmark.var()
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