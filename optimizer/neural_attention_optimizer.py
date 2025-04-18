import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
import math
from cvxopt import matrix, solvers

class OptimizerDataset(Dataset):
    """Dataset class for the neural attention optimizer"""
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class NeuralAttentionOptimizer:
    def __init__(self, 
                 sequence_length: int = 120,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 learning_rate: float = 0.0005,
                 num_epochs: int = 200,
                 batch_size: int = 64,
                 nhead: int = 8,
                 weight_decay: float = 0.02,
                 early_stopping_patience: int = 15,
                 volatility_target: float = 0.20,
                 downside_risk_weight: float = 0.5,
                 num_regimes: int = 9,
                 device: str = None):
        """Initialize the Neural Attention Optimizer
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Size of hidden layers
            num_layers: Number of transformer layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            nhead: Number of attention heads
            weight_decay: L2 regularization parameter
            early_stopping_patience: Patience for early stopping
            volatility_target: Target portfolio volatility
            downside_risk_weight: Weight for downside risk in loss
            num_regimes: Number of market regimes (default: 9)
            device: Device to use for computation (cuda/cpu)
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.nhead = nhead
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.volatility_target = volatility_target
        self.downside_risk_weight = downside_risk_weight
        self.num_regimes = num_regimes
        
        # Set device
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model components
        self.scaler = StandardScaler()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def calculate_portfolio_metrics(self, returns: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio metrics including returns, volatility, and downside risk
        
        Args:
            returns: Array of asset returns
            weights: Array of portfolio weights
            
        Returns:
            Dictionary containing portfolio metrics
        """
        portfolio_returns = np.sum(returns * weights, axis=1)
        
        # Annualization factors (assuming daily data)
        annual_factor = 252
        
        # Calculate annualized metrics
        ann_return = np.mean(portfolio_returns) * annual_factor
        ann_vol = np.std(portfolio_returns) * np.sqrt(annual_factor)
        
        # Risk-free rate (assuming 2% annual)
        risk_free_rate = 0.02
        daily_rf = risk_free_rate / annual_factor
        
        # Calculate excess returns
        excess_returns = portfolio_returns - daily_rf
        ann_excess_return = np.mean(excess_returns) * annual_factor
        
        # Calculate proper annualized Sharpe ratio
        sharpe = ann_excess_return / ann_vol if ann_vol > 0 else 0
        
        metrics = {
            'return': ann_return,  # Annualized return
            'volatility': ann_vol,  # Annualized volatility
            'sharpe': sharpe,      # Annualized Sharpe ratio
            'max_drawdown': np.max(np.maximum.accumulate(portfolio_returns) - portfolio_returns),
            'downside_risk': np.sqrt(np.mean(np.minimum(portfolio_returns - daily_rf, 0) ** 2)) * np.sqrt(annual_factor)
        }
        
        return metrics
        
    def risk_aware_loss(self, 
                       pred_weights: torch.Tensor, 
                       target_weights: torch.Tensor,
                       returns: torch.Tensor) -> torch.Tensor:
        """Calculate simplified loss focusing on MVO weights"""
        # MSE loss for tracking target weights (MVO weights)
        mse_loss = nn.MSELoss()(pred_weights, target_weights)
        
        # Calculate portfolio returns
        portfolio_returns = torch.sum(returns * pred_weights, dim=1)
        
        # Basic volatility penalty (reduced weight)
        vol = torch.std(portfolio_returns)
        vol_penalty = 0.1 * torch.abs(vol - self.volatility_target)
        
        # Expected return with momentum factor
        rolling_returns = torch.mean(returns[-20:], dim=0)
        momentum_score = torch.sum(pred_weights * rolling_returns)
        exp_return = -torch.mean(portfolio_returns) - 0.5 * momentum_score
        
        # Combine components with focus on MVO weights
        total_loss = (0.7 * mse_loss +  # Increased weight on MVO target
                     0.1 * vol_penalty +  # Reduced volatility penalty
                     1.0 * exp_return)  # Return maximization
        
        return total_loss
        
    def prepare_sequences(self, returns: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training
        
        Args:
            returns: Array or DataFrame of historical returns
            
        Returns:
            Tuple of (sequences, target_weights)
        """
        # Convert to numpy array if DataFrame
        if isinstance(returns, pd.DataFrame):
            returns = returns.values
            
        n_samples = len(returns) - self.sequence_length
        sequences = np.zeros((n_samples, self.sequence_length, returns.shape[1]))
        target_weights = np.zeros((n_samples, returns.shape[1]))
        
        for i in range(n_samples):
            sequence = returns[i:i+self.sequence_length]
            sequences[i] = sequence
            
            # Calculate target weights using mean-variance optimization
            target_weights[i] = self._mean_variance_optimization(sequence)
            
        return sequences, target_weights
        
    def _mean_variance_optimization(self, returns: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
        """Solve mean-variance optimization problem using quadratic programming"""
        n_assets = returns.shape[1]
        
        # Calculate mean returns and covariance
        mu = np.mean(returns, axis=0)
        sigma = np.cov(returns.T)
        
        # Add momentum factor
        momentum_returns = returns[-20:] if len(returns) >= 20 else returns
        momentum = np.mean(momentum_returns, axis=0)
        mu = 0.7 * mu + 0.3 * momentum  # Increased momentum weight
        
        # QP matrices
        P = matrix(2.0 * risk_aversion * sigma)
        q = matrix(-mu)
        
        # Basic constraints: weights sum to 1 and non-negative
        G = matrix(-np.eye(n_assets))
        h = matrix(np.zeros(n_assets))
        A = matrix(np.ones((1, n_assets)))
        b = matrix(np.ones(1))
        
        try:
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            if solution['status'] != 'optimal':
                return np.ones(n_assets) / n_assets
            weights = np.array(solution['x']).flatten()
        except:
            weights = np.ones(n_assets) / n_assets
            
        return weights
        
    def train(self, returns: Union[np.ndarray, pd.DataFrame], valid_returns: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Dict[str, list]:
        """Train the neural attention optimizer
        
        Args:
            returns: Training returns data
            valid_returns: Optional validation returns data
            
        Returns:
            Dictionary containing training history
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(returns, pd.DataFrame):
            returns = returns.values
        if valid_returns is not None and isinstance(valid_returns, pd.DataFrame):
            valid_returns = valid_returns.values
            
        # Prepare sequences
        X_train, y_train = self.prepare_sequences(returns)
        if valid_returns is not None:
            X_valid, y_valid = self.prepare_sequences(valid_returns)
        
        # Scale features
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        X_train = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        
        if valid_returns is not None:
            X_valid = self.scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
        
        # Create datasets
        train_dataset = OptimizerDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if valid_returns is not None:
            valid_dataset = OptimizerDataset(X_valid, y_valid)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size)
        
        # Initialize model
        self.model = NeuralAttentionModel(
            input_size=X_train.shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            nhead=self.nhead,
            sequence_length=self.sequence_length,
            num_regimes=self.num_regimes
        ).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'valid_loss': []}
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                pred_weights = self.model(batch_X)
                loss = self.risk_aware_loss(pred_weights, batch_y, batch_X[:, -1])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if valid_returns is not None:
                self.model.eval()
                valid_losses = []
                
                with torch.no_grad():
                    for batch_X, batch_y in valid_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        pred_weights = self.model(batch_X)
                        loss = self.risk_aware_loss(pred_weights, batch_y, batch_X[:, -1])
                        valid_losses.append(loss.item())
                
                avg_valid_loss = np.mean(valid_losses)
                history['valid_loss'].append(avg_valid_loss)
                
                # Early stopping and learning rate scheduling
                if avg_valid_loss < best_loss:
                    best_loss = avg_valid_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
                
                self.scheduler.step(avg_valid_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print(f"Train Loss: {avg_train_loss:.6f}")
                if valid_returns is not None:
                    print(f"Valid Loss: {avg_valid_loss:.6f}")
        
        # Load best model if validation was used
        if valid_returns is not None:
            self.model.load_state_dict(torch.load('best_model.pth'))
        
        return history
        
    def optimize_weights(self, returns: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate optimal weights for the current period
        
        Args:
            returns: Recent returns data
            
        Returns:
            Array of optimal portfolio weights
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating weights")
            
        # Convert to numpy array if DataFrame
        if isinstance(returns, pd.DataFrame):
            returns = returns.values
            
        # Prepare sequence
        sequence = returns[-self.sequence_length:]
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence = sequence_scaled.reshape(1, sequence.shape[0], sequence.shape[1])
        sequence = torch.FloatTensor(sequence).to(self.device)
        
        # Generate weights
        self.model.eval()
        with torch.no_grad():
            weights = self.model(sequence)
            weights = weights.cpu().numpy().flatten()
            
        # Ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        return weights

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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, nhead: int, sequence_length: int, num_regimes: int = 9):
        super().__init__()
        
        # Store parameters
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_regimes = num_regimes
        
        # Enhanced input embedding with multiple layers
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout, sequence_length)
        
        # Bidirectional LSTM with increased capacity
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 4  # *4 due to increased hidden size and bidirectional
        
        # Multiple attention layers with enhanced feed-forward networks
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(lstm_output_size, nhead, dropout),
                'norm1': nn.LayerNorm(lstm_output_size),
                'ffn': nn.Sequential(
                    nn.Linear(lstm_output_size, lstm_output_size * 2),
                    nn.LayerNorm(lstm_output_size * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(lstm_output_size * 2, lstm_output_size),
                    nn.LayerNorm(lstm_output_size)
                ),
                'norm2': nn.LayerNorm(lstm_output_size)
            })
            for _ in range(num_layers)
        ])
        
        # Regime-aware attention
        self.regime_attention = nn.Sequential(
            nn.Linear(lstm_output_size, num_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Regime-specific output layers
        self.regime_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size),
                nn.LayerNorm(lstm_output_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, input_size)
            )
            for _ in range(num_regimes)
        ])
        
        # Initialize weights with improved scaling
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    # Use Kaiming initialization for ReLU layers
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    # Improved uniform initialization for biases
                    bound = 1 / math.sqrt(param.shape[0])
                    nn.init.uniform_(param, -bound, bound)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, hidden_size]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # LSTM layer
        x, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*4]
        
        # Multiple attention layers with residual connections and layer normalization
        for layer in self.attention_layers:
            # Self-attention with pre-norm
            x_norm = layer['norm1'](x)
            attn_output = layer['attention'](x_norm, x_norm, x_norm)
            x = x + attn_output
            
            # Feed-forward network with pre-norm
            x_norm = layer['norm2'](x)
            ffn_output = layer['ffn'](x_norm)
            x = x + ffn_output
        
        # Attention-weighted pooling
        attention_weights = torch.softmax(
            torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(x.size(-1)), 
            dim=-1
        )
        x = torch.matmul(attention_weights, x)
        
        # Global average pooling with attention weights
        x = torch.mean(x, dim=1)  # [batch_size, hidden_size*4]
        
        # Regime attention weights
        regime_weights = self.regime_attention(x)  # [batch_size, num_regimes]
        
        # Regime-specific outputs
        regime_outputs = []
        for i in range(self.num_regimes):
            regime_output = self.regime_outputs[i](x)  # [batch_size, input_size]
            regime_outputs.append(regime_output.unsqueeze(1))
        
        regime_outputs = torch.cat(regime_outputs, dim=1)  # [batch_size, num_regimes, input_size]
        
        # Combine regime outputs using attention weights
        final_output = torch.sum(regime_outputs * regime_weights.unsqueeze(-1), dim=1)
        
        # Apply softmax to ensure weights sum to 1 and are positive
        final_output = torch.softmax(final_output, dim=-1)
        
        return final_output

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
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x) 