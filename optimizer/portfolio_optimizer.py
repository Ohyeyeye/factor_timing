import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import torch
import torch.nn as nn

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

class NeuralPortfolioOptimizer:
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
        
    def optimize_weights(self, historical_returns: pd.DataFrame) -> pd.Series:
        """Use neural network to predict optimal weights"""
        # Calculate expected returns and covariance
        expected_returns = historical_returns.mean()
        covariance = historical_returns.cov()
        
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
