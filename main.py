import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

from data.data_loader import DataLoader
from models.regime_classifier import LSTMRegimeClassifier, XGBoostRegimeClassifier, HMMRegimeClassifier
from optimization.portfolio_optimizer import MeanVarianceOptimizer, NeuralPortfolioOptimizer, RegimeAwareOptimizer
from backtest.backtester import Backtester

class FactorTimingStrategy:
    def __init__(self,
                 train_start_date: str = '2014-01-01',
                 train_end_date: str = '2019-12-31',
                 test_start_date: str = '2020-01-01',
                 test_end_date: str = '2024-12-31',
                 model_type: str = 'lstm',
                 optimizer_type: str = 'mean_variance',
                 data_dir: str = None):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing FactorTimingStrategy with {model_type} model and {optimizer_type} optimizer")
        
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        
        # Initialize components
        self.data_loader = DataLoader(data_dir=data_dir)
        self.data_loader.set_date_ranges(
            train_start_date, train_end_date,
            test_start_date, test_end_date
        )
        self.regime_classifier = self._init_regime_classifier(model_type)
        self.portfolio_optimizer = self._init_portfolio_optimizer(optimizer_type)
        self.backtester = None
        
    def _init_regime_classifier(self, model_type: str):
        """Initialize regime classifier based on model type"""
        if model_type == 'lstm':
            return LSTMRegimeClassifier(input_size=6)  # Match actual number of features
        elif model_type == 'xgboost':
            return XGBoostRegimeClassifier()
        elif model_type == 'hmm':
            return HMMRegimeClassifier()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def _init_portfolio_optimizer(self, optimizer_type: str):
        """Initialize portfolio optimizer based on type"""
        if optimizer_type == 'mean_variance':
            return MeanVarianceOptimizer()
        elif optimizer_type == 'neural':
            return NeuralPortfolioOptimizer(input_size=6)  # Adjust based on features
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training and testing data"""
        # Load Fama-French factors for both periods
        train_factors = self.data_loader.load_fama_french_factors(
            self.train_start_date, self.train_end_date)
        test_factors = self.data_loader.load_fama_french_factors(
            self.test_start_date, self.test_end_date)
            
        # Drop the 'date' column if it exists and convert returns to numeric
        if 'date' in train_factors.columns:
            train_factors = train_factors.drop('date', axis=1)
        if 'date' in test_factors.columns:
            test_factors = test_factors.drop('date', axis=1)
            
        # Convert returns to numeric, excluding the index
        numeric_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        train_factors[numeric_columns] = train_factors[numeric_columns].apply(pd.to_numeric, errors='coerce')
        test_factors[numeric_columns] = test_factors[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        # Handle missing values in factor data
        train_factors = train_factors.ffill().bfill()
        test_factors = test_factors.ffill().bfill()
            
        # Load macro and market data for both periods
        train_macro = self.data_loader.load_macro_data(
            self.train_start_date, self.train_end_date)
        test_macro = self.data_loader.load_macro_data(
            self.test_start_date, self.test_end_date)
            
        train_market = self.data_loader.load_market_data(
            self.train_start_date, self.train_end_date)
        test_market = self.data_loader.load_market_data(
            self.test_start_date, self.test_end_date)
            
        # Prepare training data for regime classification
        train_X, train_y = self.data_loader.prepare_training_data()
        
        # Store data
        self.train_data = {
            'factors': train_factors,
            'macro': train_macro,
            'market': train_market,
            'X': train_X,
            'y': train_y
        }
        
        self.test_data = {
            'factors': test_factors,
            'macro': test_macro,
            'market': test_market
        }
        
        # Ensure all data is properly aligned
        common_train_idx = train_factors.index.intersection(train_macro.index).intersection(train_market.index)
        common_test_idx = test_factors.index.intersection(test_macro.index).intersection(test_market.index)
        
        train_factors = train_factors.loc[common_train_idx]
        test_factors = test_factors.loc[common_test_idx]
        
        return train_factors, test_factors
        
    def train_models(self):
        """Train regime classifier and portfolio optimizer"""
        # Train regime classifier on training data
        self.regime_classifier.train(self.train_data['X'], self.train_data['y'])
        
        # If using neural portfolio optimizer, train it
        if isinstance(self.portfolio_optimizer, NeuralPortfolioOptimizer):
            # TODO: Prepare training data for neural portfolio optimizer
            pass
            
    def run_strategy(self) -> Dict:
        """Run the factor timing strategy with train-test split"""
        self.logger.info("Starting strategy execution...")
        
        # Prepare data
        self.logger.info("Preparing training and testing data...")
        train_factors, test_factors = self.prepare_data()
        self.logger.info(f"Prepared data - Train shape: {train_factors.shape}, Test shape: {test_factors.shape}")
        
        # Train models
        self.logger.info("Training models...")
        self.train_models()
        self.logger.info("Model training completed")
        
        # Initialize portfolio weights DataFrame for test period
        self.logger.info("Initializing portfolio weights...")
        weights = pd.DataFrame(index=test_factors.index,
                             columns=test_factors.columns)
        
        # Get regime predictions for test period
        self.logger.info("Generating regime predictions for test period...")
        test_X = pd.concat([self.test_data['macro'], self.test_data['market']], axis=1)
        # Align test_X with test_factors
        test_X = test_X.reindex(test_factors.index).ffill().bfill()
        self.logger.info(f"Test features shape after alignment: {test_X.shape}")
        regime_predictions = self.regime_classifier.predict(test_X)
        self.logger.info(f"Generated {len(regime_predictions)} regime predictions")
        
        # Calculate regime-specific returns from training period
        self.logger.info("Calculating regime-specific returns...")
        train_X = pd.concat([self.train_data['macro'], self.train_data['market']], axis=1)
        # Align train_X with train_factors
        train_X = train_X.reindex(train_factors.index).ffill().bfill()
        train_regime_predictions = self.regime_classifier.predict(train_X)
        
        # Create a DataFrame with regime predictions
        train_regimes = pd.Series(train_regime_predictions, index=train_factors.index)
        
        # Calculate regime-specific returns
        regime_returns = {}
        unique_regimes = np.unique(train_regime_predictions)
        self.logger.info(f"Found {len(unique_regimes)} unique regimes")
        
        for regime in unique_regimes:
            # Align regime mask with factor data
            regime_mask = train_regimes == regime
            # Convert to float before storing
            regime_returns[regime] = train_factors[regime_mask].apply(pd.to_numeric, errors='coerce')
            self.logger.info(f"Regime {regime}: {len(regime_returns[regime])} samples")
        
        # Create test period regime predictions series
        test_regimes = pd.Series(regime_predictions, index=test_factors.index)
        
        # Optimize weights for each period in test set
        self.logger.info("Optimizing portfolio weights...")
        optimization_count = 0
        
        for date in weights.index:
            # Get the most recent regime prediction
            current_regime = test_regimes.loc[date]
            
            # Get factor returns for the current date
            current_returns = test_factors.loc[date].astype(float)
            
            if isinstance(self.portfolio_optimizer, RegimeAwareOptimizer):
                weights.loc[date] = self.portfolio_optimizer.optimize_weights(
                    current_returns,
                    test_factors.astype(float).cov(),
                    current_regime,
                    regime_returns
                )
            else:
                weights.loc[date] = self.portfolio_optimizer.optimize_weights(
                    current_returns,
                    test_factors.astype(float).cov()
                )
            
            optimization_count += 1
            if optimization_count % 50 == 0:
                self.logger.info(f"Optimized weights for {optimization_count} periods")
                
        self.logger.info("Portfolio optimization completed")
        
        # Run backtest on test period
        self.logger.info("Running backtest...")
        self.backtester = Backtester(test_factors.astype(float))
        results = self.backtester.run_backtest(weights)
        self.logger.info("Backtest completed")
        
        return results
        
    def plot_results(self, benchmark: Optional[pd.Series] = None):
        """Plot strategy results"""
        if self.backtester is None:
            raise ValueError("Run strategy first before plotting results")
        self.backtester.plot_results(benchmark)
        
    def evaluate_strategy(self, benchmark: Optional[pd.Series] = None) -> Dict:
        """Evaluate strategy performance"""
        if self.backtester is None:
            raise ValueError("Run strategy first before evaluating")
            
        # Get strategy metrics
        strategy_metrics = self.backtester.calculate_performance_metrics()
        
        # Compare with benchmark if provided
        if benchmark is not None:
            comparison = self.backtester.compare_with_benchmark(benchmark)
            strategy_metrics.update(comparison)
            
        return strategy_metrics

def main():
    # Example usage with train-test split
    strategy = FactorTimingStrategy(
        train_start_date='2014-01-01',
        train_end_date='2019-12-31',
        test_start_date='2020-01-01',
        test_end_date='2024-12-31',
        model_type='lstm',
        optimizer_type='mean_variance',
        data_dir='data'  # Specify the data directory
    )
    
    # Run strategy
    print("Running strategy...")
    results = strategy.run_strategy()
    print("Strategy Results:", results)
    
    # Plot results
    strategy.plot_results()
    
    # Evaluate strategy
    print("Evaluating strategy...")
    evaluation = strategy.evaluate_strategy()
    print("Strategy Evaluation:", evaluation)

if __name__ == "__main__":
    main() 