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
        
    def train(self, historical_returns: pd.DataFrame):
        """Train the optimizer on historical data"""
        self.logger.info("Training mean-variance optimizer...")
        # Store historical data for reference
        self.historical_returns = historical_returns
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

class RegimeAwareOptimizer:
    def __init__(self, risk_aversion: float = 1.0, min_weight: float = 0.01):
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.logger = logging.getLogger(__name__)
        self.regime_weights = {}
        
    def train(self, historical_returns: pd.DataFrame, regime_predictions: np.ndarray, regime_returns: Dict):
        """Train the optimizer on historical data and regime information"""
        self.logger.info("Training regime-aware optimizer...")
        self.historical_returns = historical_returns
        self.regime_returns = regime_returns
        
        # Calculate optimal weights for each regime using training data
        for regime, returns in regime_returns.items():
            if len(returns) > 0:
                expected_returns = returns.mean()
                cov_matrix = returns.cov()
                weights = self._optimize_regime_weights(expected_returns, cov_matrix)
                self.regime_weights[regime] = weights
            else:
                self.logger.warning(f"No data available for regime {regime}, using equal weights")
                self.regime_weights[regime] = pd.Series(1/len(historical_returns.columns), 
                                                      index=historical_returns.columns)
        
        self.logger.info("Regime-aware optimizer training completed")
        
    def _optimize_regime_weights(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
        """Optimize weights for a specific regime"""
        try:
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
        
    def optimize_weights(self, historical_returns: pd.DataFrame, current_regime: int, regime_returns: Dict) -> pd.Series:
        """Optimize portfolio weights based on current regime and historical data"""
        if current_regime in self.regime_weights:
            weights = self.regime_weights[current_regime]
            # Ensure weights are non-negative and above minimum
            weights = weights.clip(lower=self.min_weight)
            # Normalize weights to sum to 1
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                self.logger.warning("All weights are zero, using equal weights")
                n_assets = len(historical_returns.columns)
                weights = pd.Series(1/n_assets, index=historical_returns.columns)
            return weights
        else:
            self.logger.warning(f"Unknown regime {current_regime}, using equal weights")
            n_assets = len(historical_returns.columns)
            return pd.Series(1/n_assets, index=historical_returns.columns) 
        
class AutoencoderPortfolioOptimizer(BasePortfolioOptimizer):
    def __init__(self, input_size: int = 5, hidden_size: int = 32, latent_dim: int = 3, learning_rate: float = 0.001):
        self.model = AutoencoderWeightModel(input_size, hidden_size, latent_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.logger = logging.getLogger(__name__)
        
    def train(self, historical_returns: pd.DataFrame):
        """Train the autoencoder model using historical returns"""
        self.logger.info("Training autoencoder optimizer...")
        
        # Convert data to tensor
        X = torch.FloatTensor(historical_returns.values)
        
        # Initialize target weights (equal weights)
        n_assets = historical_returns.shape[1]
        target_weights = torch.ones(n_assets) / n_assets
        
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            weights = self.model(X)
            weights = torch.softmax(weights, dim=1)  # Ensure weights sum to 1
            
            # Calculate loss (MSE between predicted weights and equal weights)
            loss = self.loss_fn(weights.mean(dim=0), target_weights)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        self.logger.info("Autoencoder optimizer training completed")
    
    def optimize_weights(self, historical_returns: pd.DataFrame) -> pd.Series:
        """Generate portfolio weights using the autoencoder"""
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor
            X = torch.FloatTensor(historical_returns.values)
            
            # Get weights from model
            weights = self.model(X)
            weights = torch.softmax(weights[-1], dim=0)  # Use last timestep's weights
            
            # Convert to series
            weights = pd.Series(weights.numpy(), index=historical_returns.columns)
            
            # Ensure weights are valid
            weights = weights.clip(lower=0)
            weights = weights / weights.sum()
            
            return weights

class AutoencoderWeightModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        # Decode
        weights = self.decoder(z)
        return weights