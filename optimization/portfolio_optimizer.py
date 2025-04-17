import numpy as np
import pandas as pd
from typing import Dict, Optional
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

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
        self.risk_aversion = risk_aversion
        
    def optimize_weights(self,
                        expected_returns: pd.Series,
                        covariance: pd.DataFrame,
                        constraints: Optional[Dict] = None) -> pd.Series:
        """Mean-variance optimization with constraints"""
        n_assets = len(expected_returns)
        w = cp.Variable(n_assets)
        
        # Define objective
        risk = cp.quad_form(w, covariance.values)
        ret = expected_returns.values @ w
        objective = cp.Maximize(ret - self.risk_aversion * risk)
        
        # Define constraints
        constraints_list = [cp.sum(w) == 1, w >= 0]  # Basic constraints
        
        # Add custom constraints if provided
        if constraints:
            if 'max_weight' in constraints:
                constraints_list.append(w <= constraints['max_weight'])
            if 'min_weight' in constraints:
                constraints_list.append(w >= constraints['min_weight'])
                
        # Solve optimization problem
        prob = cp.Problem(objective, constraints_list)
        prob.solve()
        
        return pd.Series(w.value, index=expected_returns.index)

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
        # Prepare input features
        features = pd.concat([
            expected_returns,
            pd.Series(np.diag(covariance), index=expected_returns.index)  # Volatilities
        ], axis=1)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features.values)
        
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