import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

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
                 optimizer_type: str = 'mean_variance'):
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        
        # Initialize components
        self.data_loader = DataLoader()
        self.regime_classifier = self._init_regime_classifier(model_type)
        self.portfolio_optimizer = self._init_portfolio_optimizer(optimizer_type)
        self.backtester = None
        
    def _init_regime_classifier(self, model_type: str):
        """Initialize regime classifier based on model type"""
        if model_type == 'lstm':
            return LSTMRegimeClassifier(input_size=8)  # Adjust input_size based on features
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
        # Prepare data
        train_factors, test_factors = self.prepare_data()
        
        # Train models
        self.train_models()
        
        # Initialize portfolio weights DataFrame for test period
        weights = pd.DataFrame(index=test_factors.index,
                             columns=test_factors.columns)
        
        # Get regime predictions for test period
        test_X = pd.concat([self.test_data['macro'], self.test_data['market']], axis=1)
        regime_predictions = self.regime_classifier.predict(test_X)
        
        # Calculate regime-specific returns from training period
        train_regime_predictions = self.regime_classifier.predict(self.train_data['X'])
        regime_returns = {}
        for regime in np.unique(train_regime_predictions):
            regime_mask = train_regime_predictions == regime
            regime_returns[regime] = self.train_data['factors'][regime_mask]
            
        # Optimize weights for each period in test set
        for date in weights.index:
            current_regime = regime_predictions[date]
            
            if isinstance(self.portfolio_optimizer, RegimeAwareOptimizer):
                weights.loc[date] = self.portfolio_optimizer.optimize_weights(
                    test_factors.loc[date],
                    test_factors.cov(),
                    current_regime,
                    regime_returns
                )
            else:
                weights.loc[date] = self.portfolio_optimizer.optimize_weights(
                    test_factors.loc[date],
                    test_factors.cov()
                )
                
        # Run backtest on test period
        self.backtester = Backtester(test_factors)
        results = self.backtester.run_backtest(weights)
        
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
        optimizer_type='mean_variance'
    )
    
    # Run strategy
    results = strategy.run_strategy()
    print("Strategy Results:", results)
    
    # Plot results
    strategy.plot_results()
    
    # Evaluate strategy
    evaluation = strategy.evaluate_strategy()
    print("Strategy Evaluation:", evaluation)

if __name__ == "__main__":
    main() 