import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
import math
import os

# Add safe globals for numpy types
from torch.serialization import add_safe_globals
add_safe_globals([np._core.multiarray.scalar, np.dtype])

class NeuralAttentionOptimizer:
    def __init__(self, sequence_length: int = 60, 
                 hidden_size: int = 64, num_layers: int = 2, 
                 nhead: int = 8, dropout: float = 0.2,
                 learning_rate: float = 0.001, num_epochs: int = 100, 
                 batch_size: int = 32, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.model_path = model_path
        
    def save_model(self, path: str = None):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        save_path = path or self.model_path
        if save_path is None:
            raise ValueError("No path specified for saving the model")
            
        # Create model state dictionary including all necessary components
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_embedding.in_features,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'nhead': self.nhead,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length
            },
            'scaler_state': self.scaler.__dict__,
            'sequence_length': self.sequence_length
        }
        
        # Save the model state
        try:
            # First try to remove existing file to ensure clean save
            if os.path.exists(save_path):
                os.remove(save_path)
            torch.save(model_state, save_path, _use_new_zipfile_serialization=True)
            self.logger.info(f"Model saved successfully to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, path: str = None):
        """Load a trained model and scaler"""
        load_path = path or self.model_path
        if load_path is None:
            raise ValueError("No path specified for loading the model")
            
        try:
            # Load the model state with weights_only=False
            model_state = torch.load(load_path, map_location=self.device, weights_only=False)
            
            # Reconstruct the model
            model_config = model_state['model_config']
            self.model = NeuralAttentionModel(
                input_size=model_config['input_size'],
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                nhead=model_config['nhead'],
                sequence_length=model_config['sequence_length']
            ).to(self.device)
            
            # Load the model state dict
            self.model.load_state_dict(model_state['model_state_dict'])
            
            # Load the scaler state
            self.scaler.__dict__.update(model_state['scaler_state'])
            
            # Update instance variables
            self.sequence_length = model_state['sequence_length']
            
            self.logger.info(f"Model loaded successfully from {load_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
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
    
    def prepare_sequences(self, returns: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences of past returns and corresponding target weights"""
        sequences = []
        targets = []
        
        # Convert returns to numpy array
        returns_array = returns.values
        
        for t in range(self.sequence_length, len(returns)):
            # Get past returns
            past_returns = returns_array[t-self.sequence_length:t]
            
            # Calculate target weights using MVO
            target_weights = self._mean_variance_optimization(
                returns_array[:t],
                risk_aversion=1.0
            )
            
            sequences.append(past_returns)
            targets.append(target_weights)
        
        # Convert lists to numpy arrays before creating tensors
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)
    
    def train(self, returns: pd.DataFrame):
        """Train the optimizer"""
        self.logger.info("Training neural attention optimizer...")
        
        # Check if model already exists
        if self.model_path and os.path.exists(self.model_path):
            self.logger.info("Found existing model, loading...")
            self.load_model()
            return
        
        # Prepare sequences and targets
        sequences, targets = self.prepare_sequences(returns)
        
        # Scale the sequences
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = self.scaler.fit_transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(sequences.shape)
        
        # Create dataset and dataloader
        dataset = OptimizerDataset(sequences_scaled, targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = NeuralAttentionModel(
            input_size=sequences.shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            nhead=self.nhead,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_sequences, batch_targets in dataloader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
        
        # Save the trained model
        if self.model_path:
            self.save_model()
            
        self.logger.info("Training completed successfully")
    
    def optimize_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Optimize weights based on recent factor returns"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get the most recent sequence
        recent_returns = returns.iloc[-self.sequence_length:]
        
        # Scale the sequence
        recent_returns_scaled = self.scaler.transform(recent_returns)
        recent_returns_scaled = torch.FloatTensor(recent_returns_scaled).unsqueeze(0).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            weights = self.model(recent_returns_scaled)
            weights = weights.squeeze().cpu().numpy()
            
        # Ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return weights

class OptimizerDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output

class NeuralAttentionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, nhead: int, sequence_length: int):
        super().__init__()
        
        # Store sequence length
        self.sequence_length = sequence_length
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size, nhead, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Sequence reduction (reduce sequence dimension to 1)
        self.sequence_reduction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output = nn.Sequential(
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
        self.sequence_reduction[0].weight.data.uniform_(-initrange, initrange)
        self.output[0].weight.data.uniform_(-initrange, initrange)
        self.output[3].weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)
        
        # Self-attention
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        # Reduce sequence dimension by taking mean
        x = x.mean(dim=1)
        
        # Apply sequence reduction
        x = self.sequence_reduction(x)
        
        # Output layer
        x = self.output(x)
        
        # Apply softmax to ensure weights sum to 1
        x = torch.softmax(x, dim=-1)
        
        return x 