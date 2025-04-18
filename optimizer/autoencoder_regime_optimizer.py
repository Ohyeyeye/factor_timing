import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
import logging

class Autoencoder(nn.Module):
    def __init__(self, input_size: int, encoding_dim: int):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

class AutoencoderRegimeOptimizer:
    def __init__(self, n_regimes: int = 3, input_size: int = 5, encoding_dim: int = 2):
        self.n_regimes = n_regimes
        self.input_size = input_size  # Number of factors
        self.encoding_dim = encoding_dim  # Dimension of the encoded representation
        self.autoencoders = {}  # One autoencoder per regime
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, returns: pd.DataFrame, regime_predictions: np.ndarray):
        """Train a separate autoencoder for each regime"""
        self.logger.info("Training autoencoder regime optimizer...")
        
        # Convert returns to numpy array
        returns_array = returns.values
        
        for regime in range(self.n_regimes):
            regime_mask = regime_predictions == regime
            if np.sum(regime_mask) > 0:
                self.logger.info(f"Training autoencoder for regime {regime} with {np.sum(regime_mask)} samples")
                regime_returns = returns_array[regime_mask]
                
                # Build and train autoencoder for this regime
                self.autoencoders[regime] = self._build_and_train_autoencoder(regime_returns)
            else:
                self.logger.warning(f"No samples found for regime {regime}")
                
    def _build_and_train_autoencoder(self, returns: np.ndarray):
        """Build and train an autoencoder for a specific regime"""
        model = Autoencoder(self.input_size, self.encoding_dim).to(self.device)
        optimizer = optim.Adam(model.parameters())
        
        # Convert data to PyTorch tensors
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
        
            # Forward pass
            reconstructed = model(returns_tensor)
            
            # Calculate reconstruction loss
            reconstruction_loss = nn.MSELoss()(reconstructed, returns_tensor)
            
            # Calculate portfolio optimization loss
            portfolio_returns = torch.sum(reconstructed * returns_tensor, dim=1)
            returns_variance = torch.var(returns_tensor, dim=0)
            portfolio_variance = torch.sum(
                torch.matmul(reconstructed, torch.matmul(torch.diag(returns_variance), reconstructed.t()))
            )
            sharpe_loss = -torch.mean(portfolio_returns) / (torch.sqrt(portfolio_variance) + 1e-6)
            
            # Combine losses
            loss = reconstruction_loss + 0.1 * sharpe_loss
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        return model
        
    def optimize_weights(self, returns: pd.DataFrame, current_regime: int) -> np.ndarray:
        """Optimize weights based on current regime using autoencoder"""
        if current_regime in self.autoencoders:
            # Get the most recent returns
            recent_returns = returns.iloc[-1].values.reshape(1, -1)
            recent_returns_tensor = torch.FloatTensor(recent_returns).to(self.device)
            
            # Get the autoencoder for this regime
            model = self.autoencoders[current_regime]
            model.eval()
            
            with torch.no_grad():
                # Get the encoded representation and decode it to weights
                encoded = model.encode(recent_returns_tensor)
                weights = model.decode(encoded).cpu().numpy()[0]
            
            # Ensure weights sum to 1 and are non-negative
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
            return weights
        else:
            # Fallback to equal weights if no autoencoder exists for this regime
            self.logger.warning(f"No autoencoder found for regime {current_regime}, using equal weights")
            return np.ones(self.input_size) / self.input_size
            
    def get_regime_stats(self) -> Dict:
        """Get statistics for each regime's autoencoder"""
        stats = {}
        for regime, model in self.autoencoders.items():
            stats[regime] = {
                'model_summary': str(model),
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'encoding_dim': self.encoding_dim
            }
        return stats 