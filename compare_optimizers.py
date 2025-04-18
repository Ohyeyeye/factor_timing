import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from data.data_loader import DataLoader
from classifier.regime_classifier import LSTMRegimeClassifier
from optimizer.regime_aware_optimizer import TransformerRegimeOptimizer
from backtest.backtester import Backtester

class OptimizerComparison:
    def __init__(self,
                 train_start_date: str = '2014-01-01',
                 train_end_date: str = '2019-12-31',
                 test_start_date: str = '2020-01-01',
                 test_end_date: str = '2024-12-31',
                 data_dir: str = 'data',
                 model_dir: str = 'models'):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize data loader
        self.data_loader = DataLoader(data_dir=data_dir)
        self.data_loader.set_date_ranges(
            train_start_date, train_end_date,
            test_start_date, test_end_date
        )
        
        # Initialize regime classifier
        self.regime_classifier = LSTMRegimeClassifier(
            n_regimes=5,
            sequence_length=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            num_epochs=100,
            batch_size=32,
            model_path=os.path.join(model_dir, 'lstm_classifier.pth')
        )
        
        # Initialize optimizers
        self.lstm_optimizer = TransformerRegimeOptimizer(
            n_regimes=5,
            sequence_length=60,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            num_epochs=100,
            batch_size=32
        )
        
        self.transformer_optimizer = TransformerRegimeOptimizer(
            n_regimes=5,
            sequence_length=60,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            num_epochs=100,
            batch_size=32,
            nhead=8
        )
        
        # Initialize backtester (will be set with returns later)
        self.backtester = None
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and testing data"""
        self.logger.info("Preparing data...")
        
        # Load Fama-French factors
        train_factors = self.data_loader.load_fama_french_factors(
            self.data_loader.train_start_date,
            self.data_loader.train_end_date
        )
        test_factors = self.data_loader.load_fama_french_factors(
            self.data_loader.test_start_date,
            self.data_loader.test_end_date
        )
        
        # Convert percentage returns to decimal
        train_factors = train_factors / 100
        test_factors = test_factors / 100
        
        # Prepare regime classification data
        train_X, train_y = self.data_loader.prepare_training_data()
        test_X, _ = self.data_loader.prepare_training_data(is_test=True)
        
        # Train regime classifier
        self.logger.info("Training regime classifier...")
        self.regime_classifier.train(train_X, train_y)
        
        # Get regime predictions
        train_regimes = self.regime_classifier.predict(train_X)
        test_regimes = self.regime_classifier.predict(test_X)
        
        return train_factors, test_factors, train_regimes, test_regimes
        
    def train_optimizers(self, train_factors: pd.DataFrame, train_regimes: np.ndarray):
        """Train both optimizers"""
        self.logger.info("Training LSTM optimizer...")
        self.lstm_optimizer.train(train_factors, pd.Series(train_regimes))
        
        self.logger.info("Training Transformer optimizer...")
        self.transformer_optimizer.train(train_factors, pd.Series(train_regimes))
        
    def run_backtest(self, test_factors: pd.DataFrame, test_regimes: np.ndarray) -> Dict[str, Dict]:
        """Run backtest for both optimizers"""
        self.logger.info("Running backtests...")
        
        results = {}
        
        # Initialize weight DataFrames
        lstm_weights = pd.DataFrame(index=test_factors.index,
                                  columns=test_factors.columns)
        transformer_weights = pd.DataFrame(index=test_factors.index,
                                         columns=test_factors.columns)
        
        # Generate weights for each period
        for date in test_factors.index:
            current_regime = test_regimes[date]
            historical_returns = test_factors.loc[:date]
            
            # LSTM weights
            lstm_weights.loc[date] = self.lstm_optimizer.optimize_weights(
                historical_returns, current_regime
            )
            
            # Transformer weights
            transformer_weights.loc[date] = self.transformer_optimizer.optimize_weights(
                historical_returns, current_regime
            )
        
        # Initialize backtester with test returns
        self.backtester = Backtester(test_factors)
        
        # Run backtests
        lstm_results = self.backtester.run_backtest(test_factors, lstm_weights)
        transformer_results = self.backtester.run_backtest(test_factors, transformer_weights)
        
        # Add regime-specific analysis
        lstm_results['regime_analysis'] = self._analyze_regime_performance(
            lstm_results['portfolio_returns'], test_regimes
        )
        transformer_results['regime_analysis'] = self._analyze_regime_performance(
            transformer_results['portfolio_returns'], test_regimes
        )
        
        results['lstm'] = lstm_results
        results['transformer'] = transformer_results
        
        return results
        
    def _analyze_regime_performance(self, returns: pd.Series, regimes: np.ndarray) -> Dict:
        """Analyze performance by regime"""
        regime_returns = {}
        for regime in range(5):
            regime_mask = regimes == regime
            if regime_mask.any():
                regime_returns[regime] = returns[regime_mask]
        
        analysis = {}
        for regime, returns in regime_returns.items():
            analysis[regime] = {
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(12),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns)
            }
        return analysis
        
    def plot_comparison(self, results: Dict[str, Dict]):
        """Plot comparison of results"""
        self.logger.info("Plotting comparison...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Plot cumulative returns
        lstm_returns = results['lstm']['cumulative_returns']
        transformer_returns = results['transformer']['cumulative_returns']
        
        axes[0, 0].plot(lstm_returns.index, lstm_returns.values, label='LSTM')
        axes[0, 0].plot(transformer_returns.index, transformer_returns.values, label='Transformer')
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot drawdowns
        lstm_drawdown = results['lstm']['drawdown']
        transformer_drawdown = results['transformer']['drawdown']
        
        axes[0, 1].plot(lstm_drawdown.index, lstm_drawdown.values, label='LSTM')
        axes[0, 1].plot(transformer_drawdown.index, transformer_drawdown.values, label='Transformer')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot monthly returns distribution
        lstm_monthly = results['lstm']['monthly_returns']
        transformer_monthly = results['transformer']['monthly_returns']
        
        sns.histplot(lstm_monthly, ax=axes[1, 0], label='LSTM', alpha=0.5)
        sns.histplot(transformer_monthly, ax=axes[1, 0], label='Transformer', alpha=0.5)
        axes[1, 0].set_title('Monthly Returns Distribution')
        axes[1, 0].legend()
        
        # Plot weight turnover
        lstm_turnover = results['lstm']['turnover']
        transformer_turnover = results['transformer']['turnover']
        
        axes[1, 1].plot(lstm_turnover.index, lstm_turnover.values, label='LSTM')
        axes[1, 1].plot(transformer_turnover.index, transformer_turnover.values, label='Transformer')
        axes[1, 1].set_title('Weight Turnover')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot regime-specific Sharpe ratios
        regimes = range(5)
        lstm_sharpe = [results['lstm']['regime_analysis'][r]['sharpe_ratio'] for r in regimes]
        transformer_sharpe = [results['transformer']['regime_analysis'][r]['sharpe_ratio'] for r in regimes]
        
        x = np.arange(len(regimes))
        width = 0.35
        axes[2, 0].bar(x - width/2, lstm_sharpe, width, label='LSTM')
        axes[2, 0].bar(x + width/2, transformer_sharpe, width, label='Transformer')
        axes[2, 0].set_title('Sharpe Ratio by Regime')
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels([f'Regime {r}' for r in regimes])
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Plot regime-specific returns
        lstm_returns = [results['lstm']['regime_analysis'][r]['mean_return'] for r in regimes]
        transformer_returns = [results['transformer']['regime_analysis'][r]['mean_return'] for r in regimes]
        
        axes[2, 1].bar(x - width/2, lstm_returns, width, label='LSTM')
        axes[2, 1].bar(x + width/2, transformer_returns, width, label='Transformer')
        axes[2, 1].set_title('Mean Return by Regime')
        axes[2, 1].set_xticks(x)
        axes[2, 1].set_xticklabels([f'Regime {r}' for r in regimes])
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('optimizer_comparison.png')
        plt.close()
        
    def print_metrics(self, results: Dict[str, Dict]):
        """Print performance metrics comparison"""
        print("\nPerformance Metrics Comparison:")
        print("-------------------------------")
        print(f"{'Metric':<20} {'LSTM':<15} {'Transformer':<15}")
        print("-" * 50)
        
        metrics = [
            ('Total Return', lambda x: f"{x['total_return']:.2%}"),
            ('Annualized Return', lambda x: f"{x['annualized_return']:.2%}"),
            ('Annualized Volatility', lambda x: f"{x['annualized_volatility']:.2%}"),
            ('Sharpe Ratio', lambda x: f"{x['sharpe_ratio']:.2f}"),
            ('Max Drawdown', lambda x: f"{x['max_drawdown']:.2%}"),
            ('Annualized Turnover', lambda x: f"{x['turnover']:.2%}")
        ]
        
        for metric_name, formatter in metrics:
            lstm_value = formatter(results['lstm'])
            transformer_value = formatter(results['transformer'])
            print(f"{metric_name:<20} {lstm_value:<15} {transformer_value:<15}")
            
        # Print regime-specific metrics
        print("\nRegime-Specific Performance:")
        print("---------------------------")
        for regime in range(5):
            print(f"\nRegime {regime}:")
            print(f"{'Metric':<20} {'LSTM':<15} {'Transformer':<15}")
            print("-" * 50)
            
            regime_metrics = [
                ('Mean Return', lambda x: f"{x['regime_analysis'][regime]['mean_return']:.2%}"),
                ('Volatility', lambda x: f"{x['regime_analysis'][regime]['volatility']:.2%}"),
                ('Sharpe Ratio', lambda x: f"{x['regime_analysis'][regime]['sharpe_ratio']:.2f}"),
                ('Skewness', lambda x: f"{x['regime_analysis'][regime]['skewness']:.2f}"),
                ('Kurtosis', lambda x: f"{x['regime_analysis'][regime]['kurtosis']:.2f}")
            ]
            
            for metric_name, formatter in regime_metrics:
                lstm_value = formatter(results['lstm'])
                transformer_value = formatter(results['transformer'])
                print(f"{metric_name:<20} {lstm_value:<15} {transformer_value:<15}")
            
    def run_comparison(self):
        """Run full comparison"""
        self.logger.info("Starting optimizer comparison...")
        
        # Prepare data
        train_factors, test_factors, train_regimes, test_regimes = self.prepare_data()
        
        # Train optimizers
        self.train_optimizers(train_factors, train_regimes)
        
        # Run backtests
        results = self.run_backtest(test_factors, test_regimes)
        
        # Plot comparison
        self.plot_comparison(results)
        
        # Print metrics
        self.print_metrics(results)
        
        self.logger.info("Comparison completed")

def main():
    comparison = OptimizerComparison()
    comparison.run_comparison()

if __name__ == "__main__":
    main() 