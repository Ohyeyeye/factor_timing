import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
import logging

class RegimeModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.network(x)

class NeuralRegimeOptimizer:
    def __init__(self, n_regimes: int = 3, input_size: int = 5, hidden_size: int = 64):
        self.n_regimes = n_regimes
        self.input_size = input_size  # Number of factors
        self.hidden_size = hidden_size
        self.models = {}  # One model per regime
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, returns: pd.DataFrame, regime_predictions: np.ndarray):
        """Train a separate neural network for each regime"""
        self.logger.info("Training neural regime optimizer...")
        
        # Convert returns to numpy array
        returns_array = returns.values
        
        for regime in range(self.n_regimes):
            regime_mask = regime_predictions == regime
            if np.sum(regime_mask) > 0:
                self.logger.info(f"Training model for regime {regime} with {np.sum(regime_mask)} samples")
                regime_returns = returns_array[regime_mask]
                
                # Build and train model for this regime
                self.models[regime] = self._build_and_train_model(regime_returns)
            else:
                self.logger.warning(f"No samples found for regime {regime}")
                
    def _build_and_train_model(self, returns: np.ndarray):
        """Build and train a neural network model for a specific regime"""
        model = RegimeModel(self.input_size, self.hidden_size).to(self.device)
        optimizer = optim.Adam(model.parameters())
        
        # Convert data to PyTorch tensors
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
        
            # Forward pass
            weights = model(returns_tensor)
            
            # Calculate portfolio returns
            portfolio_returns = torch.sum(weights * returns_tensor, dim=1)
            
            # Calculate portfolio variance
            returns_variance = torch.var(returns_tensor, dim=0)
            portfolio_variance = torch.sum(
                torch.matmul(weights, torch.matmul(torch.diag(returns_variance), weights.t()))
            )
            
            # Custom loss: negative Sharpe ratio-like metric
            loss = -torch.mean(portfolio_returns) / (torch.sqrt(portfolio_variance) + 1e-6)
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        return model
        
    def optimize_weights(self, returns: pd.DataFrame, current_regime: int) -> np.ndarray:
        """Optimize weights based on current regime"""
        if current_regime in self.models:
            # Get the most recent returns
            recent_returns = returns.iloc[-1].values.reshape(1, -1)
            recent_returns_tensor = torch.FloatTensor(recent_returns).to(self.device)
            
            # Predict optimal weights using the regime-specific model
            model = self.models[current_regime]
            model.eval()
            with torch.no_grad():
                weights = model(recent_returns_tensor).cpu().numpy()[0]
            
            # Ensure weights sum to 1 and are non-negative
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
            return weights
        else:
            # Fallback to equal weights if no model exists for this regime
            self.logger.warning(f"No model found for regime {current_regime}, using equal weights")
            return np.ones(self.input_size) / self.input_size
            
    def get_regime_stats(self) -> Dict:
        """Get statistics for each regime's model"""
        stats = {}
        for regime, model in self.models.items():
            stats[regime] = {
                'model_summary': str(model),
                'num_parameters': sum(p.numel() for p in model.parameters())
            }
        return stats 