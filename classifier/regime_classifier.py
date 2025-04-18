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
import torch.optim as optim

class BaseRegimeClassifier:
    def __init__(self, n_regimes=5):
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
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMRegimeClassifier(BaseRegimeClassifier):
    def __init__(self, n_regimes: int = 5, sequence_length: int = 20, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2, learning_rate: float = 0.001,
                 num_epochs: int = 100, batch_size: int = 32):
        super().__init__(n_regimes)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_sequences(self, X: pd.DataFrame) -> torch.Tensor:
        """Prepare sequences for LSTM by handling missing values and scaling features."""
        # Handle missing values
        X = X.copy()
        X = X.ffill().bfill().fillna(0)
        
        # Scale features
        if not hasattr(self, 'scaler_fitted'):
            self.scaler.fit(X)
            self.scaler_fitted = True
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        sequences = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            sequences.append(X_scaled[i:i + self.sequence_length])
        
        if not sequences:
            raise ValueError("Not enough data points to create sequences")
            
        return torch.FloatTensor(np.array(sequences))
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the LSTM model on the given data."""
        try:
            # Prepare sequences
            X_seq = self.prepare_sequences(X)
            y = torch.LongTensor(y.values[self.sequence_length-1:])
            
            # Initialize model
            input_size = X.shape[1]
            self.model = LSTMModel(input_size, self.hidden_size, self.num_layers, 
                                 self.n_regimes, self.dropout)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(self.num_epochs):
                # Mini-batch training
                for i in range(0, len(X_seq), self.batch_size):
                    batch_X = X_seq[i:i + self.batch_size]
                    batch_y = y[i:i + self.batch_size]
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    logging.info(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
                    
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels for new data
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted regime labels
        """
        self.logger.info("Predicting regime labels...")
        
        # Ensure test data has same features as training data
        if not hasattr(self, 'feature_names'):
            raise ValueError("Model must be trained before prediction")
        
        # Reorder columns to match training data and fill missing features with 0
        X = X.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale features
        if not hasattr(self, 'scaler'):
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X)
        
        # Prepare sequences
        X_sequences = self.prepare_sequences(X_scaled)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_sequences)
            _, predicted = torch.max(outputs.data, 1)
            
        self.logger.info(f"Predicted regime distribution: {dict(pd.Series(predicted.numpy()).value_counts())}")
        
        return predicted.numpy()

class XGBoostRegimeClassifier(BaseRegimeClassifier):
    def __init__(self, **params):
        super().__init__()
        from xgboost import XGBClassifier
        # Ensure num_class is set correctly
        if 'num_class' not in params:
            params['num_class'] = self.n_regimes
        if 'objective' not in params:
            params['objective'] = 'multi:softmax'
        if 'eval_metric' not in params:
            params['eval_metric'] = 'mlogloss'
        self.model = XGBClassifier(**params)
        self.logger.info(f"Initialized XGBoostRegimeClassifier with {self.n_regimes} regimes")
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train XGBoost model"""
        self.logger.info("Training XGBoost model...")
        X_scaled = self.prepare_features(X)
        # Ensure labels are integers starting from 0
        y = y.astype(int)
        self.model.fit(X_scaled, y)
        self.logger.info("XGBoost model training completed")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime using trained XGBoost"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return self._validate_predictions(predictions)

class HMMRegimeClassifier(BaseRegimeClassifier):
    def __init__(self, n_components=5, n_iter=100, random_state=42):
        super().__init__(n_regimes=n_components)
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.logger.info(f"Initialized HMMRegimeClassifier with {n_components} regimes")
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train HMM model"""
        self.logger.info("Training HMM model...")
        
        # Handle missing values
        X = X.ffill().bfill().fillna(0)
        
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
        """Predict regime using trained HMM"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Handle missing values
        X = X.ffill().bfill().fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get regime predictions
        predictions = self.model.predict(X_scaled)
        return self._validate_predictions(predictions) 