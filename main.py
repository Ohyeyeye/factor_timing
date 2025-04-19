import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging
import os

from data.data_loader import DataLoader
from predictor.regime_predictor import LSTMRegimePredictor
from optimizer.neural_attention_optimizer import NeuralAttentionOptimizer
from backtest.backtester import Backtester

class FactorTimingStrategy:
    def __init__(self,
                 train_start_date: str = '2014-01-01',
                 train_end_date: str = '2019-12-31',
                 test_start_date: str = '2020-01-01',
                 test_end_date: str = '2024-12-31',
                 model_type: str = 'lstm',
                 optimizer_type: str = 'regime_aware',
                 data_dir: str = None,
                 model_dir: str = 'models',
                 load_pretrained: bool = True):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing FactorTimingStrategy with {model_type} model and {optimizer_type} optimizer")
        
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.model_dir = model_dir
        self.model_type = model_type
        self.load_pretrained = load_pretrained
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize factor columns
        self.factor_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        
        # Initialize components
        self.data_loader = DataLoader(data_dir=data_dir)
        self.data_loader.set_date_ranges(
            train_start_date, train_end_date,
            test_start_date, test_end_date
        )
        
        # Set model paths with absolute paths
        self.model_paths = {
            'lstm': os.path.join(os.path.abspath(model_dir), 'lstm_classifier.pth'),
            'xgboost': os.path.join(os.path.abspath(model_dir), 'xgboost_classifier.pkl'),
            'hmm': os.path.join(os.path.abspath(model_dir), 'hmm_classifier.pkl')
        }
        
        self.logger.info(f"Model will be saved to/loaded from: {self.model_paths[model_type]}")
        
        self.regime_classifier = self._init_regime_classifier(model_type)
        self.portfolio_optimizer = self._init_portfolio_optimizer(optimizer_type)
        self.backtester = None
        
    def _init_regime_classifier(self, model_type: str):
        """Initialize regime classifier based on model type"""
        model_path = self.model_paths[model_type]
        self.logger.info(f"Initializing {model_type} classifier. Model path: {model_path}")
        
        if model_type == 'lstm':
            classifier = LSTMRegimePredictor(
                n_regimes=5,
                sequence_length=20,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                learning_rate=0.001,
                num_epochs=100,
                batch_size=32,
                model_path=model_path
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Try to load pretrained model if requested
        if self.load_pretrained and os.path.exists(model_path):
            try:
                self.logger.info(f"Loading pretrained model from {model_path}")
                if model_type == 'lstm':
                    classifier.load_model()
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            except Exception as e:
                self.logger.warning(f"Failed to load pretrained model: {str(e)}")
                self.logger.info("Will train a new model")
        
        return classifier
            
    def _init_portfolio_optimizer(self, optimizer_type: str):
        """Initialize portfolio optimizer based on type"""
        if optimizer_type == 'neural_attention':
            model_path = os.path.join(self.model_dir, 'neural_attention_optimizer.pth')
            return NeuralAttentionOptimizer(
                sequence_length=60,
                hidden_size=64,
                num_layers=2,
                nhead=8,
                dropout=0.2,
                learning_rate=0.001,
                num_epochs=100,
                batch_size=32,
                model_path=model_path
            )
        else:
            self.logger.warning(f"Unknown optimizer type: {optimizer_type}")
            return None
            
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training and testing data"""
        self.logger.info("Starting data preparation...")
        
        # Load Fama-French factors for both periods
        self.logger.info("Loading Fama-French factors...")
        train_factors = self.data_loader.load_fama_french_factors(
            self.train_start_date, self.train_end_date)
        test_factors = self.data_loader.load_fama_french_factors(
            self.test_start_date, self.test_end_date)
        self.logger.info("Fama-French factors loaded")
        
        # Drop the 'date' column if it exists and convert returns to numeric
        self.logger.info("Processing factor data...")
        if 'date' in train_factors.columns:
            train_factors = train_factors.drop('date', axis=1)
        if 'date' in test_factors.columns:
            test_factors = test_factors.drop('date', axis=1)

        # Convert returns to numeric and select factor columns
        factor_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        train_factors = train_factors[factor_columns].apply(pd.to_numeric, errors='coerce')
        test_factors = test_factors[factor_columns].apply(pd.to_numeric, errors='coerce')

        # Convert percentage returns to decimal format (e.g., -0.87% -> -0.0087)
        train_factors = train_factors / 100
        test_factors = test_factors / 100

        # Handle missing values in factor data
        train_factors = train_factors.ffill().bfill()
        test_factors = test_factors.ffill().bfill()
        self.logger.info("Factor data processed")

        # Log some statistics about the factor returns
        self.logger.info("\nFactor Returns Summary (in decimal):")
        self.logger.info("\nTraining Period:")
        self.logger.info(train_factors.describe())
        self.logger.info("\nTesting Period:")
        self.logger.info(test_factors.describe())

        # Load macro and market data for both periods
        self.logger.info("Loading macro and market data...")
        train_macro = self.data_loader.load_macro_data(
            self.train_start_date, self.train_end_date)
        test_macro = self.data_loader.load_macro_data(
            self.test_start_date, self.test_end_date)

        train_market = self.data_loader.load_market_data(
            self.train_start_date, self.train_end_date)
        test_market = self.data_loader.load_market_data(
            self.test_start_date, self.test_end_date)
        self.logger.info("Macro and market data loaded")

        # Prepare training data for regime classification
        self.logger.info("Preparing training data for regime classification...")
        train_X, train_y = self.data_loader.prepare_training_data()
        
        # Prepare test data for regime classification
        self.logger.info("Preparing test data for regime classification...")
        test_X, test_y = self.data_loader.prepare_training_data(is_test=True)
        self.logger.info("Test data prepared")

        # Store data
        self.logger.info("Storing data...")
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
            'market': test_market,
            'X': test_X,
            'y': test_y
        }

        # Ensure all data is properly aligned
        self.logger.info("Aligning data indices...")
        common_train_idx = train_factors.index.intersection(train_macro.index).intersection(train_market.index)
        common_test_idx = test_factors.index.intersection(test_macro.index).intersection(test_market.index)

        train_factors = train_factors.loc[common_train_idx]
        test_factors = test_factors.loc[common_test_idx]
        self.logger.info("Data indices aligned")

        self.logger.info("Data preparation completed")
        return train_factors, test_factors
        
    def train_models(self):
        """Train regime classifier and portfolio optimizer"""
        self.logger.info("Starting model training...")
        
        # Get training data
        train_X = self.train_data['X']
        train_y = self.train_data['y']
        train_factors = self.train_data['factors']
        
        # Check if we need to train
        model_path = self.model_paths[self.model_type]
        model_exists = os.path.exists(model_path)
        self.logger.info(f"Checking for existing model at: {model_path}")
        self.logger.info(f"Model exists: {model_exists}, Load pretrained: {self.load_pretrained}")
        
        if self.load_pretrained and model_exists:
            try:
                if self.model_type == 'lstm':
                    self.logger.info("Loading pretrained LSTM model...")
                    self.regime_classifier.load_model(model_path)
                elif self.model_type == 'xgboost':
                    self.logger.info("Loading pretrained XGBoost model...")
                    self.regime_classifier.load_model()
                self.logger.info("Successfully loaded pretrained model")
                train_pred = self.regime_classifier.predict(train_X)
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                self.logger.info("Falling back to training new model...")
                self._train_new_model(train_X, train_y)
        else:
            if not model_exists:
                self.logger.info("No existing model found")
            elif not self.load_pretrained:
                self.logger.info("Load pretrained is disabled")
            self._train_new_model(train_X, train_y)
            
        # Train portfolio optimizer
        self.logger.info("Training portfolio optimizer...")
        self.portfolio_optimizer.train(train_factors)
        self.logger.info("Portfolio optimizer training completed")
        
    def _train_new_model(self, train_X, train_y):
        """Train a new model from scratch"""
        self.logger.info("Training new model...")
        # Train the model
        self.regime_classifier.train(train_X, train_y)
        train_pred = self.regime_classifier.predict(train_X)
        
        # Save the trained model
        try:
            if self.model_type == 'lstm':
                self.regime_classifier.save_model()
            elif self.model_type == 'xgboost':
                self.regime_classifier.save_model()
            self.logger.info(f"Model saved to {self.model_paths[self.model_type]}")
        except Exception as e:
            self.logger.warning(f"Failed to save model: {str(e)}")
            
        self._evaluate_model(train_y, train_pred)
        
    def _evaluate_model(self, train_y, train_pred):
        """Evaluate model performance and create visualizations"""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Training metrics
        train_accuracy = accuracy_score(train_y, train_pred)
        
        print("\nModel Training Results:")
        print("----------------------")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        print("\nTraining Set Classification Report:")
        print(classification_report(train_y, train_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        train_cm = confusion_matrix(train_y, train_pred)
        sns.heatmap(train_cm, annot=True, fmt='d')
        plt.title('Training Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_results.png'))
        plt.close()
        
        return {
            'train_accuracy': train_accuracy,
            'train_confusion_matrix': train_cm
        }
        
    def run_strategy(self) -> Dict:
        """Run the factor timing strategy with train-test split"""
        self.logger.info("Starting strategy execution...")
        
        # Prepare data
        self.logger.info("Preparing training and testing data...")
        train_factors, test_factors = self.prepare_data()
        self.logger.info(f"Prepared data - Train shape: {train_factors.shape}, Test shape: {test_factors.shape}")
        
        # Train models and get metrics
        self.logger.info("Training models...")
        training_metrics = self.train_models()
        
        # Display additional training insights
        print("\nRegime Distribution in Training Data:")
        regime_dist = pd.Series(self.train_data['y']).value_counts(normalize=True)
        print(regime_dist)
        
        print("\nFeature Importance:")
        if hasattr(self.regime_classifier, 'feature_importance'):
            feature_importance = pd.Series(
                self.regime_classifier.feature_importance(),
                index=self.train_data['X'].columns
            ).sort_values(ascending=False)
            print(feature_importance)
        
        self.logger.info("Model training completed")
        
        # Initialize portfolio weights DataFrame for test period
        self.logger.info("Initializing portfolio weights...")
        weights = pd.DataFrame(index=test_factors.index,
                             columns=test_factors.columns)
        self.logger.info("Portfolio weights initialized")
        
        # Get regime predictions for test period
        self.logger.info("Generating regime predictions for test period...")
        test_X, _ = self.data_loader.prepare_training_data(is_test=True)
        test_X = test_X.reindex(test_factors.index).ffill().bfill()
        self.logger.info(f"Test features shape after alignment: {test_X.shape}")
        regime_predictions = self.regime_classifier.predict(test_X)
        self.logger.info(f"Generated {len(regime_predictions)} regime predictions")
        
        # Create a Series with the same index as test_factors
        regime_predictions_series = pd.Series(regime_predictions, index=test_factors.index)
        
        # Optimize weights for each period in test set
        self.logger.info("Optimizing portfolio weights...")
        optimization_count = 0
        
        for date in weights.index:
            # Get the most recent regime prediction
            current_regime = regime_predictions_series[date]
            
            # Get historical returns up to the current date
            historical_returns = test_factors.loc[:date].astype(float)
            
            weights.loc[date] = self.portfolio_optimizer.optimize_weights(
                historical_returns
            )
            
            optimization_count += 1
            if optimization_count % 50 == 0:
                self.logger.info(f"Optimized weights for {optimization_count} periods")
                
        self.logger.info("Portfolio optimization completed")
        
        # Initialize backtester with test period returns
        self.logger.info("Initializing backtester...")
        self.backtester = Backtester(test_factors.astype(float))
        
        # Run backtest and get results
        self.logger.info("Running backtest...")
        self.backtest_results = self.backtester.run_backtest(test_factors.astype(float), weights)
        self.logger.info("Backtest completed")
        
        # Print performance metrics
        print("\nStrategy Performance Metrics:")
        print("----------------------------")
        print(f"Total Return: {self.backtest_results['total_return']:.2%}")
        print(f"Annualized Return: {self.backtest_results['annualized_return']:.2%}")
        print(f"Annualized Volatility: {self.backtest_results['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.backtest_results['max_drawdown']:.2%}")
        print(f"Annualized Turnover: {self.backtest_results['turnover']:.2%}")
        
        self.logger.info("Strategy execution completed")
        return self.backtest_results
        
    def plot_results(self, benchmarks: Optional[Dict[str, pd.Series]] = None):
        """Plot backtest results with multiple benchmarks"""
        self.logger.info("Starting to plot results...")
        
        # Ensure backtest is run first
        if self.backtester is None:
            self.logger.info("Running backtest first...")
            self.run_strategy()
        
        # Now plot the results
        self.backtester.plot_results(benchmarks)
        self.logger.info("Plot completed")
        
    def evaluate_strategy(self, benchmark: Optional[pd.Series] = None) -> Dict:
        """Evaluate strategy performance"""
        self.logger.info("Starting strategy evaluation...")
        if self.backtester is None:
            raise ValueError("Run strategy first before evaluating")
            
        # Get strategy metrics
        self.logger.info("Calculating performance metrics...")
        strategy_metrics = self.backtester.calculate_performance_metrics(self.backtester.portfolio_returns)
        
        # Compare with benchmark if provided
        if benchmark is not None:
            self.logger.info("Comparing with benchmark...")
            comparison = self.backtester.compare_with_benchmark(benchmark)
            strategy_metrics.update(comparison)
            
        self.logger.info("Strategy evaluation completed")
        return strategy_metrics

def main():
    # Multiple train-test splits tested manually:
    # 1. Long training: 2000-2019 train, 2020-2024 eval
    # 2. Medium training: 2014-2019 train, 2020-2024 backtest
    # 3. Extended training: 2000-2021 train, 2022-2024 backtest
    # 4. Short training: 2020-2021 train, 2022-2023 backtest
    strategy = FactorTimingStrategy(
        train_start_date='2000-01-01',
        train_end_date='2019-12-31',
        test_start_date='2020-01-01',
        test_end_date='2024-12-31',
        model_type='lstm',
        optimizer_type='neural_attention',
        data_dir='data'
    )
    
    # Create benchmarks
    print("Creating benchmarks...")
    
    # Load Fama-French factors for test period
    ff5_factors = strategy.data_loader.load_fama_french_factors(
        strategy.test_start_date, 
        strategy.test_end_date
    )
    
    # Convert percentage returns to decimal
    ff5_factors = ff5_factors.apply(pd.to_numeric, errors='coerce') / 100
    
    print("FF5 factors loaded and converted to decimal format")
    
    # Calculate equal weight returns (excluding RF)
    factor_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    equal_weight_returns = ff5_factors[factor_columns].mean(axis=1)
    
    # Load S&P 500 returns
    sp500_returns = strategy.data_loader.load_market_data(
        strategy.test_start_date,
        strategy.test_end_date
    )['SP500_Return']
    
    # Create benchmark dictionary
    benchmarks = {
        'Equal Weight FF5': equal_weight_returns,
        'S&P 500': sp500_returns
    }

    # Run strategy
    print("Running strategy...")
    results = strategy.run_strategy()
    print("Strategy Results:", results)
    
    # Plot results with both benchmarks
    strategy.plot_results(benchmarks)
    
    # Evaluate strategy against both benchmarks
    print("Evaluating strategy...")
    evaluation = strategy.evaluate_strategy(benchmarks)
    print("Strategy Evaluation:", evaluation)

if __name__ == "__main__":
    main() 