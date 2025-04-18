import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
import math

class TransformerRegimeOptimizer:
    def __init__(self, n_regimes: int = 5, sequence_length: int = 60, 
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2,
                 learning_rate: float = 0.001, num_epochs: int = 100, batch_size: int = 32,
                 nhead: int = 8):
        self.logger = logging.getLogger(__name__)
        self.n_regimes = n_regimes
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.nhead = nhead
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.regime_models = {}  # One model per regime
        
    def _mean_variance_optimization(self, returns: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
        """Calculate optimal weights using mean-variance optimization"""
        # Calculate expected returns and covariance matrix
        expected_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns, rowvar=False)
        
        # Use quadratic programming to find optimal weights
        n_assets = len(expected_returns)
        A = np.ones((1, n_assets))
        b = np.array([1.0])
        G = -np.eye(n_assets)
        h = np.zeros(n_assets)
        
        # Solve quadratic programming problem
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False
        P = matrix(cov_matrix * risk_aversion)
        q = matrix(-expected_returns)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        
        solution = solvers.qp(P, q, G, h, A, b)
        weights = np.array(solution['x']).flatten()
        
        return weights
    
    def prepare_sequences(self, returns: pd.DataFrame, regimes: Union[pd.Series, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences of past returns and corresponding target weights"""
        sequences = []
        targets = []
        
        # Convert returns to numpy array
        returns_array = returns.values
        
        # Convert regimes to numpy array if it's a pandas Series
        if isinstance(regimes, pd.Series):
            regimes = regimes.values
            
        for t in range(self.sequence_length, len(returns)):
            # Get past returns
            past_returns = returns_array[t-self.sequence_length:t]
            
            # Get current regime
            current_regime = regimes[t]
            
            # Calculate optimal weights using MVO
            optimal_weights = self._mean_variance_optimization(
                returns_array[:t],
                risk_aversion=1.0
            )
            
            sequences.append(past_returns)
            targets.append(optimal_weights)
            
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)
    
    def train(self, returns: pd.DataFrame, regimes: pd.Series):
        """Train the optimizer"""
        self.logger.info("Training regime-aware optimizer...")
        
        # Prepare sequences and targets
        sequences, targets = self.prepare_sequences(returns, regimes)
        
        # Scale the sequences
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = self.scaler.fit_transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(sequences.shape)
        
        # Create dataset and dataloader
        dataset = RegimeOptimizerDataset(sequences_scaled, targets, regimes[self.sequence_length:])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model for each regime
        for regime in range(self.n_regimes):
            self.logger.info(f"Training model for regime {regime}...")
            
            # Create model for this regime
            model = TransformerOptimizerModel(
                input_size=sequences.shape[-1],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                nhead=self.nhead
            ).to(self.device)
            
            # Initialize optimizer and loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            for epoch in range(self.num_epochs):
                model.train()
                total_loss = 0
                
                for batch_sequences, batch_targets, batch_regimes in dataloader:
                    # Only train on samples from current regime
                    mask = (batch_regimes == regime)
                    if not mask.any():
                        continue
                        
                    batch_sequences = batch_sequences[mask].to(self.device)
                    batch_targets = batch_targets[mask].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_sequences)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Regime {regime} - Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
            
            self.regime_models[regime] = model
        
        self.logger.info("Training completed successfully")
    
    def optimize_weights(self, returns: pd.DataFrame, current_regime: int) -> np.ndarray:
        """Optimize weights based on current regime and factor returns"""
        if current_regime not in self.regime_models:
            raise ValueError(f"No model trained for regime {current_regime}")
            
        # Get the most recent sequence
        recent_returns = returns.iloc[-self.sequence_length:]
        
        # Scale the sequence
        recent_returns_scaled = self.scaler.transform(recent_returns)
        recent_returns_scaled = torch.FloatTensor(recent_returns_scaled).unsqueeze(0).to(self.device)
        
        # Get model for current regime
        model = self.regime_models[current_regime]
        model.eval()
        
        with torch.no_grad():
            weights = model(recent_returns_scaled)
            weights = weights.squeeze().cpu().numpy()
            
        # Ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return weights

class RegimeOptimizerDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, regimes: Union[pd.Series, np.ndarray]):
        self.sequences = sequences
        self.targets = targets
        # Convert regimes to numpy array if it's a pandas Series
        if isinstance(regimes, pd.Series):
            self.regimes = regimes.values
        else:
            self.regimes = regimes
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
            self.regimes[idx]
        )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerOptimizerModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, nhead: int):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder[0].weight.data.uniform_(-initrange, initrange)
        self.decoder[3].weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, hidden_size]
        
        # Decoder
        x = self.decoder(x)
        
        # Apply softmax to ensure weights sum to 1
        x = torch.softmax(x, dim=1)
        
        return x 