import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from hmmlearn import hmm

class BaseRegimeClassifier:
    def __init__(self, n_regimes=2):
        self.logger = logging.getLogger(__name__)
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.logger.info(f"Initialized BaseRegimeClassifier with {n_regimes} regimes")
        
    def prepare_features(self, X):
        """Prepare features for training/prediction"""
        self.logger.info("Preparing features...")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            self.logger.info(f"Input shape: {X.shape}")
            
        # Handle missing values
        if isinstance(X, pd.DataFrame):
            self.logger.info("Handling missing values...")
            missing_before = X.isnull().sum().sum()
            X = X.ffill(limit=5)  # Forward fill with limit
            X = X.bfill(limit=5)  # Backward fill with limit
            X = X.fillna(0)  # Fill any remaining NaNs with 0
            X = X.replace([np.inf, -np.inf], 0)  # Replace infinities
            missing_after = X.isnull().sum().sum()
            self.logger.info(f"Missing values: {missing_before} -> {missing_after}")
            
        # Scale features
        self.logger.info("Scaling features...")
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(X)
            self.logger.info("Fitted new scaler")
        X_scaled = self.scaler.transform(X)
        self.logger.info(f"Output shape: {X_scaled.shape}")
        
        return X_scaled
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the regime classifier"""
        raise NotImplementedError("Subclasses must implement train()")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime"""
        raise NotImplementedError("Subclasses must implement predict()")
        
    def _validate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Validate and clean predictions"""
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
        predictions = np.clip(predictions, 0, self.n_regimes - 1)
        return predictions.astype(int)

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.batch_norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return torch.sigmoid(out)

class LSTMRegimeClassifier(BaseRegimeClassifier):
    def __init__(self, 
                 input_size: int = None,  # Will be determined from data
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 batch_size: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.model = None  # Will be initialized after seeing data
        self.criterion = nn.BCELoss()
        self.optimizer = None  # Will be initialized with model
        self.logger.info(f"Initialized LSTMRegimeClassifier with {hidden_size} hidden units, {num_layers} layers")
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train LSTM model with validation split and early stopping"""
        self.logger.info("Training LSTM model...")
        
        # Determine input size from data if not specified
        if self.input_size is None:
            self.input_size = X.shape[1]
            self.logger.info(f"Set input size to {self.input_size}")
            
        # Initialize model if not already done
        if self.model is None:
            self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, 1, self.dropout)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.logger.info("Initialized LSTM model and optimizer")
            
        X_scaled = self.prepare_features(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)  # Add sequence dimension
        y_tensor = torch.FloatTensor(y.values)
        self.logger.info(f"Prepared tensors - X: {X_tensor.shape}, y: {y_tensor.shape}")
        
        # Create train-validation split
        train_size = int(0.8 * len(X_tensor))
        indices = torch.randperm(len(X_tensor))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.logger.info(f"Created data loaders - Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Best)")
            else:
                patience_counter += 1
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime using trained LSTM"""
        self.logger.info("Generating LSTM predictions...")
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)
        self.logger.info(f"Input tensor shape: {X_tensor.shape}")
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs.squeeze().numpy() > 0.5).astype(int)
            
        regime_counts = np.bincount(predictions)
        self.logger.info(f"Prediction distribution: {dict(enumerate(regime_counts))}")
        
        return predictions

class XGBoostRegimeClassifier(BaseRegimeClassifier):
    def __init__(self, **params):
        super().__init__()
        from xgboost import XGBClassifier
        self.model = XGBClassifier(**params)
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train XGBoost model"""
        X_scaled = self.prepare_features(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime using trained XGBoost"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class HMMRegimeClassifier(BaseRegimeClassifier):
    def __init__(self, n_components=2, n_iter=100, random_state=42):
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train HMM model"""
        self.logger.info("Training HMM model...")
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        try:
            self.model.fit(X_scaled)
            self.logger.info("HMM model training completed")
        except Exception as e:
            self.logger.error(f"Error training HMM model: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regimes using trained HMM model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        try:
            return self.model.predict(X_scaled)
        except Exception as e:
            self.logger.error(f"Error predicting with HMM model: {str(e)}")
            raise 