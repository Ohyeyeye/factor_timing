import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from scipy.optimize import minimize
import logging

class BasePortfolioOptimizer:
    def __init__(self):
        pass
    
    def optimize_weights(self,
                        expected_returns: pd.Series,
                        covariance: pd.DataFrame,
                        constraints: Optional[Dict] = None) -> pd.Series:
        """Optimize portfolio weights"""
        raise NotImplementedError

class MeanVarianceOptimizer:
    def __init__(self, risk_aversion: float = 1.0, min_weight: float = 0.01):
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.logger = logging.getLogger(__name__)
        self.returns = None
        self.cov_matrix = None
        
    def train(self, historical_returns: pd.DataFrame):
        """Train the optimizer on historical data"""
        self.logger.info("Training mean-variance optimizer...")
        # Store historical data for reference
        self.returns = historical_returns
        self.cov_matrix = historical_returns.cov()
        self.logger.info("Mean-variance optimizer training completed")
        
    def optimize_weights(self, historical_returns: pd.DataFrame) -> pd.Series:
        """Optimize portfolio weights using historical data up to current date"""
        try:
            # Calculate expected returns and covariance matrix from historical data
            expected_returns = historical_returns.mean()
            cov_matrix = historical_returns.cov()
            
            # Solve mean-variance optimization
            n_assets = len(expected_returns)
            A = np.zeros((n_assets + 1, n_assets + 1))
            A[0:n_assets, 0:n_assets] = 2 * self.risk_aversion * cov_matrix
            A[n_assets, 0:n_assets] = 1
            A[0:n_assets, n_assets] = 1
            
            b = np.zeros(n_assets + 1)
            b[0:n_assets] = expected_returns
            b[n_assets] = 1
            
            weights = np.linalg.solve(A, b)[0:n_assets]
            
            # Ensure weights are non-negative and above minimum
            weights = np.maximum(weights, self.min_weight)
            
            # Normalize weights to sum to 1
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                self.logger.warning("All weights are zero, using equal weights")
                weights = np.ones(n_assets) / n_assets
                
            weights = pd.Series(weights, index=expected_returns.index)
            return weights
            
        except (np.linalg.LinAlgError, ZeroDivisionError) as e:
            self.logger.warning(f"Optimization failed: {e}, using equal weights")
            return pd.Series(1/n_assets, index=expected_returns.index)

class NeuralPortfolioOptimizer(BasePortfolioOptimizer):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.model = NeuralPortfolioModel(input_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def update_input_size(self, new_input_size: int):
        """Update the input size of the neural network model"""
        self.input_size = new_input_size
        self.model = NeuralPortfolioModel(new_input_size, self.hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def optimize_weights(self,
                        expected_returns: pd.Series,
                        covariance: pd.DataFrame,
                        constraints: Optional[Dict] = None) -> pd.Series:
        """Use neural network to predict optimal weights"""
        # Convert expected returns and covariance to numeric values
        expected_returns = pd.to_numeric(expected_returns, errors='coerce')
        covariance = covariance.astype(float)
        
        # Prepare input features
        features = pd.concat([
            expected_returns,
            pd.Series(np.diag(covariance), index=expected_returns.index)  # Volatilities
        ], axis=1)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features.values.astype(float))
        
        # Get predictions
        with torch.no_grad():
            weights = self.model(features_tensor)
            weights = torch.softmax(weights, dim=0)  # Ensure weights sum to 1
            
        return pd.Series(weights.numpy(), index=expected_returns.index)
    
    def train(self,
             features: pd.DataFrame,
             target_weights: pd.DataFrame,
             num_epochs: int = 100):
        """Train the neural network"""
        features_tensor = torch.FloatTensor(features.values)
        target_tensor = torch.FloatTensor(target_weights.values)
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            predictions = self.model(features_tensor)
            loss = nn.MSELoss()(predictions, target_tensor)
            loss.backward()
            self.optimizer.step()

class NeuralPortfolioModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(NeuralPortfolioModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output one weight per asset
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RegimeAwareOptimizer(MeanVarianceOptimizer):
    def __init__(self, n_regimes: int = 3):
        super().__init__()
        self.n_regimes = n_regimes
        self.regime_optimizers = {}  # Store separate optimizers for each regime
        self.regime_returns = {}  # Store historical returns for each regime
        
    def train(self, factors: pd.DataFrame, regime_predictions: np.ndarray, regime_returns: Dict[int, pd.DataFrame] = None):
        """Train regime-specific optimizers"""
        self.logger.info("Training regime-aware optimizer...")
        
        # Store regime returns if provided
        if regime_returns is not None:
            self.regime_returns = regime_returns
        
        # Train separate optimizers for each regime
        for regime in range(self.n_regimes):
            regime_mask = regime_predictions == regime
            regime_factors = factors[regime_mask]
            
            if len(regime_factors) > 0:
                self.regime_optimizers[regime] = MeanVarianceOptimizer()
                self.regime_optimizers[regime].train(regime_factors)
                self.logger.info(f"Trained optimizer for regime {regime} with {len(regime_factors)} samples")
            else:
                self.logger.warning(f"No samples found for regime {regime}")
                
    def optimize_weights(self, factors: pd.DataFrame, current_regime: int, regime_returns: Dict[int, pd.DataFrame] = None) -> np.ndarray:
        """Optimize weights based on current regime"""
        self.logger.info(f"Optimizing weights for regime {current_regime}")
        
        # Use regime-specific optimizer if available
        if current_regime in self.regime_optimizers:
            return self.regime_optimizers[current_regime].optimize_weights(factors)
        
        # Fallback to base optimizer if regime-specific optimizer not available
        self.logger.warning(f"No optimizer found for regime {current_regime}, using base optimizer")
        return super().optimize_weights(factors)
        
    def get_regime_stats(self) -> Dict:
        """Get statistics for each regime's optimizer"""
        stats = {}
        for regime, optimizer in self.regime_optimizers.items():
            stats[regime] = {
                'n_samples': len(optimizer.returns),
                'mean_returns': optimizer.returns.mean(),
                'cov_matrix': optimizer.cov_matrix
            }
        return stats 