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

class MeanVarianceOptimizer(BasePortfolioOptimizer):
    def __init__(self, risk_aversion: float = 1.0, max_weight: float = 0.5):
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.logger = logging.getLogger(__name__)
        
    def optimize_weights(self, expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.Series:
        """Optimize portfolio weights using mean-variance optimization"""
        self.logger.info("Optimizing portfolio weights...")
        
        # Convert inputs to numpy arrays
        mu = expected_returns.values
        Sigma = covariance.values
        
        # Number of assets
        n = len(mu)
        
        # Define optimization problem
        def objective(w):
            return -(w @ mu - self.risk_aversion * w @ Sigma @ w)
            
        # Initial guess (equal weights)
        w0 = np.ones(n) / n
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w},  # Weights >= 0
            {'type': 'ineq', 'fun': lambda w: self.max_weight - w}  # Weights <= max_weight
        ]
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            self.logger.warning("Optimization did not converge, using equal weights")
            weights = np.ones(n) / n
        else:
            weights = result.x
            
        # Create Series with original index
        weights_series = pd.Series(weights, index=expected_returns.index)
        
        # Log weight distribution
        self.logger.info(f"Weight distribution: {weights_series.to_dict()}")
        
        return weights_series

class NeuralPortfolioOptimizer(BasePortfolioOptimizer):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 learning_rate: float = 0.001):
        self.model = NeuralPortfolioModel(input_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
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

class RegimeAwareOptimizer(BasePortfolioOptimizer):
    def __init__(self, base_optimizer: BasePortfolioOptimizer):
        self.base_optimizer = base_optimizer
        
    def optimize_weights(self,
                        expected_returns: pd.Series,
                        covariance: pd.DataFrame,
                        regime: int,
                        regime_returns: Dict[int, pd.DataFrame],
                        constraints: Optional[Dict] = None) -> pd.Series:
        """Optimize weights based on current regime"""
        # Get regime-specific returns
        regime_ret = regime_returns[regime]
        
        # Calculate regime-specific expected returns and covariance
        regime_exp_returns = regime_ret.mean()
        regime_cov = regime_ret.cov()
        
        # Use base optimizer with regime-specific parameters
        return self.base_optimizer.optimize_weights(
            regime_exp_returns,
            regime_cov,
            constraints
        ) 