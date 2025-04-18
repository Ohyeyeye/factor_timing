import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
import logging
from optimizer.portfolio_optimizer import MeanVarianceOptimizer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class RegimeSpecificNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_size // 2, input_size),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights using Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

class NeuralRegimeOptimizer:
    def __init__(self, n_regimes: int = 5, input_size: int = 5, hidden_size: int = 64,
                 learning_rate: float = 0.001, num_epochs: int = 200, batch_size: int = 32):
        self.n_regimes = n_regimes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create separate networks for each regime
        self.regime_networks = nn.ModuleDict({
            f'regime_{i}': RegimeSpecificNetwork(input_size, hidden_size).to(self.device)
            for i in range(n_regimes)
        })
        
        # Initialize optimizers for each network
        self.optimizers = {
            f'regime_{i}': optim.AdamW(self.regime_networks[f'regime_{i}'].parameters(),
                                     lr=learning_rate, weight_decay=0.01)
            for i in range(n_regimes)
        }
        
        # Initialize schedulers for each optimizer
        self.schedulers = {
            f'regime_{i}': optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[f'regime_{i}'],
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            for i in range(n_regimes)
        }
        
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def prepare_training_data(self, returns: pd.DataFrame, regimes: np.ndarray) -> Dict:
        """Prepare training data for each regime"""
        regime_data = {}
        
        # Convert returns to numpy and handle missing values
        returns_np = returns.fillna(0).values
        
        # Scale the returns
        scaled_returns = self.scaler.fit_transform(returns_np)
        
        # Prepare data for each regime
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            if np.sum(regime_mask) > 0:
                regime_returns = returns_np[regime_mask]
                
                # Ensure minimum number of samples
                if len(regime_returns) < 2:
                    self.logger.warning(f"Not enough samples for regime {regime}, skipping")
                    continue
                    
                # Calculate target weights using Sharpe ratio optimization
                target_weights = self.calculate_target_weights(regime_returns)
                
                # Replicate target weights for each time step
                X = torch.FloatTensor(scaled_returns[regime_mask]).to(self.device)
                y = torch.FloatTensor(np.tile(target_weights, (len(X), 1))).to(self.device)
                
                regime_data[regime] = {'X': X, 'y': y}
                self.logger.info(f"Prepared data for regime {regime}: {len(X)} samples")
                
        return regime_data
        
    def calculate_target_weights(self, returns: np.ndarray) -> np.ndarray:
        """Calculate target weights using Sharpe ratio optimization"""
        try:
            # Ensure minimum number of samples
            if len(returns) < 2:
                return np.ones(returns.shape[1]) / returns.shape[1]
            
            # Calculate mean returns and covariance with shrinkage
            mean_returns = np.mean(returns, axis=0)
            
            # Apply shrinkage to covariance matrix for stability
            sample_cov = np.cov(returns.T)
            shrinkage_factor = 0.5  # Adjust this value between 0 and 1
            target_cov = np.diag(np.diag(sample_cov))  # Diagonal matrix
            cov_matrix = shrinkage_factor * target_cov + (1 - shrinkage_factor) * sample_cov
            
            # Add small constant to diagonal for numerical stability
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-5
            
            # Calculate optimal weights using Sharpe ratio optimization
            try:
                inv_cov = np.linalg.inv(cov_matrix)
                weights = inv_cov.dot(mean_returns)
            except np.linalg.LinAlgError:
                # If matrix inversion fails, use pseudo-inverse
                inv_cov = np.linalg.pinv(cov_matrix)
                weights = inv_cov.dot(mean_returns)
            
            # Apply constraints
            weights = np.maximum(weights, 0)  # Non-negative weights
            weights_sum = np.sum(weights)
            
            if weights_sum > 0:
                weights = weights / weights_sum  # Normalize to sum to 1
            else:
                # Fallback to equal weights if optimization fails
                weights = np.ones(returns.shape[1]) / returns.shape[1]
                
        except Exception as e:
            self.logger.warning(f"Error in weight calculation: {str(e)}, using equal weights")
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            
        return weights
        
    def train(self, returns: pd.DataFrame, regimes: np.ndarray):
        """Train regime-specific networks"""
        self.logger.info("Training neural regime optimizer...")
        
        # Prepare training data for each regime
        regime_data = self.prepare_training_data(returns, regimes)
        
        if not regime_data:
            self.logger.warning("No valid regime data for training")
            return
        
        # Train networks for each regime
        for regime in range(self.n_regimes):
            if regime not in regime_data:
                continue
                
            network = self.regime_networks[f'regime_{regime}']
            optimizer = self.optimizers[f'regime_{regime}']
            scheduler = self.schedulers[f'regime_{regime}']
            
            X = regime_data[regime]['X']
            y = regime_data[regime]['y']
            
            # Verify tensor shapes
            if X.size(0) != y.size(0):
                self.logger.error(f"Shape mismatch in regime {regime}: X: {X.shape}, y: {y.shape}")
                continue
            
            # Create DataLoader for batching
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True)
            
            best_loss = float('inf')
            patience_counter = 0
            best_state = None
            
            for epoch in range(self.num_epochs):
                network.train()
                total_loss = 0
                num_batches = 0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pred_weights = network(batch_X)
                    
                    # Calculate losses
                    mse_loss = nn.MSELoss()(pred_weights, batch_y)
                    
                    # Add regularization for diversity in weights
                    diversity_penalty = -torch.mean(torch.std(pred_weights, dim=1))
                    
                    # Add constraint for minimum weight
                    min_weight_penalty = torch.mean(torch.relu(0.05 - pred_weights))
                    
                    # Combine losses
                    loss = mse_loss + 0.1 * diversity_penalty + 0.1 * min_weight_penalty
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    scheduler.step(avg_loss)
                    
                    # Early stopping
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        patience_counter = 0
                        best_state = network.state_dict()
                    else:
                        patience_counter += 1
                        if patience_counter >= 20:
                            if best_state is not None:
                                network.load_state_dict(best_state)
                            break
                            
                    if (epoch + 1) % 10 == 0:
                        self.logger.info(f"Regime {regime} - Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
                        
        self.logger.info("Training completed successfully")
        
    def optimize_weights(self, returns: pd.DataFrame, current_regime: int) -> np.ndarray:
        """Optimize portfolio weights based on current regime and returns"""
        if f'regime_{current_regime}' not in self.regime_networks:
            self.logger.warning(f"No network found for regime {current_regime}, using equal weights")
            return np.ones(returns.shape[1]) / returns.shape[1]
            
        network = self.regime_networks[f'regime_{current_regime}']
        network.eval()
        
        with torch.no_grad():
            # Prepare input data
            recent_returns = returns.iloc[-20:].fillna(0).values  # Use last 20 days
            scaled_returns = self.scaler.transform(recent_returns)
            X = torch.FloatTensor(scaled_returns).to(self.device)
            
            # Get predictions
            weights = network(X)
            
            # Average weights over the sequence
            weights = weights.mean(dim=0).cpu().numpy()
            
            # Ensure non-negative weights that sum to 1
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
        return weights
        
    def update_input_size(self, input_size: int):
        """Update input size for all networks"""
        self.input_size = input_size
        # Recreate networks with new input size
        self.regime_networks = nn.ModuleDict({
            f'regime_{i}': RegimeSpecificNetwork(input_size, self.hidden_size).to(self.device)
            for i in range(self.n_regimes)
        })
        # Reinitialize optimizers
        self.optimizers = {
            f'regime_{i}': optim.AdamW(self.regime_networks[f'regime_{i}'].parameters(),
                                     lr=self.learning_rate, weight_decay=0.01)
            for i in range(self.n_regimes)
        }

    def get_regime_stats(self) -> Dict:
        """Get statistics for each regime's model"""
        stats = {}
        for regime, model in self.regime_networks.items():
            stats[regime] = {
                'model_summary': str(model),
                'num_parameters': sum(p.numel() for p in model.parameters())
            }
        return stats 