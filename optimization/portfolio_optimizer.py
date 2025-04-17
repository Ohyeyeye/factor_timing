import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from scipy.optimize import minimize

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
    def __init__(self, risk_aversion: float = 1.0):
        """Initialize mean-variance optimizer"""
        self.risk_aversion = risk_aversion
        
    def _clean_data(self, returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Clean input data by handling NaN values"""
        # Forward fill any NaN values in returns
        clean_returns = returns.ffill().fillna(0)
        
        # Handle NaN values in covariance matrix
        clean_cov = cov_matrix.copy()
        # Fill diagonal with small positive values if NaN
        np.fill_diagonal(clean_cov.values, 
                        np.where(np.isnan(np.diag(clean_cov)), 1e-6, np.diag(clean_cov)))
        # Forward fill remaining NaN values
        clean_cov = clean_cov.ffill().bfill().fillna(0)
        
        return clean_returns, clean_cov
        
    def optimize_weights(self, 
                        returns: pd.Series, 
                        cov_matrix: pd.DataFrame,
                        **kwargs) -> np.ndarray:
        """
        Optimize portfolio weights using mean-variance optimization
        
        Args:
            returns: Expected returns for each asset
            cov_matrix: Covariance matrix of asset returns
            **kwargs: Additional arguments (unused)
            
        Returns:
            np.ndarray: Optimal portfolio weights
        """
        # Clean input data
        clean_returns, clean_cov = self._clean_data(returns, cov_matrix)
        
        n_assets = len(returns)
        
        # Optimization objective: maximize returns - risk_aversion * variance
        def objective(weights):
            portfolio_return = np.sum(weights * clean_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(clean_cov, weights)))
            return -(portfolio_return - self.risk_aversion * portfolio_risk)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # weights sum to 1
        ]
        
        # Bounds for each weight (0 to 1)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, 
                        initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        if not result.success:
            # If optimization fails, return equal weights
            self.logger.warning("Optimization failed. Returning equal weights.")
            return initial_weights
            
        return result.x

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