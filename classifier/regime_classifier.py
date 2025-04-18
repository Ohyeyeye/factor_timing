import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from hmmlearn import hmm
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import os

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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply batch normalization to input
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(-1, x.size(-1))
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers with residual connections
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        residual = out
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Add residual connection if dimensions match
        if residual.size(-1) == out.size(-1):
            out = out + residual
            
        out = self.fc3(out)
        return out

class LSTMRegimeClassifier(BaseRegimeClassifier):
    def __init__(self, n_regimes: int = 5, sequence_length: int = 20, 
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 learning_rate: float = 0.001, num_epochs: int = 200, batch_size: int = 32,
                 model_path: str = None):
        super().__init__(n_regimes)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_path = model_path
        
    def prepare_sequences(self, X: np.ndarray) -> torch.Tensor:
        """Prepare sequences for LSTM input"""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequence = X[i:(i + self.sequence_length)]
            sequences.append(sequence)
        return torch.FloatTensor(np.array(sequences)).to(self.device)
        
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
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_classes': self.n_regimes,
                'dropout': self.dropout
            },
            'scaler_state': self.scaler.__dict__,
            'sequence_length': self.sequence_length,
            'n_regimes': self.n_regimes
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
            # Load the model state with weights_only=False since we trust our own saved model
            model_state = torch.load(load_path, map_location=self.device, weights_only=False)
            
            # Reconstruct the model
            model_config = model_state['model_config']
            self.model = LSTMModel(
                input_size=model_config['input_size'],
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout']
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(model_state['model_state_dict'])
            
            # Load scaler state
            self.scaler.__dict__.update(model_state['scaler_state'])
            
            # Update other attributes
            self.sequence_length = model_state['sequence_length']
            self.n_regimes = model_state['n_regimes']
            
            self.logger.info(f"Model loaded successfully from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the LSTM model with option to save"""
        self.logger.info("Training LSTM regime classifier...")
        
        # Convert inputs to numpy arrays
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_np = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_np)
        
        # Prepare sequences
        X_seq = self.prepare_sequences(X_scaled)
        y_seq = torch.LongTensor(y_np[self.sequence_length-1:]).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_seq, y_seq)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_np),
            y=y_np
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Initialize model if not loaded
        if self.model is None:
            input_size = X_np.shape[1]
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=self.n_regimes,
                dropout=self.dropout
            ).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        # Training loop
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        self.model.eval()
        
        # Save model if path is specified
        if self.model_path:
            self.save_model()
            
        self.logger.info("Training completed successfully")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regimes using the trained LSTM model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        self.model.eval()
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X_scaled = self.scaler.transform(X_np)
        X_seq = self.prepare_sequences(X_scaled)
        
        with torch.no_grad():
            outputs = self.model(X_seq)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
        # Pad predictions to match input length
        padded_predictions = np.full(len(X), -1)
        padded_predictions[self.sequence_length-1:] = predictions
        padded_predictions[:self.sequence_length-1] = predictions[0]
        
        return self._validate_predictions(padded_predictions)
        
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance based on attention weights"""
        if self.model is None:
            return None
            
        # Extract attention weights
        attention_weights = []
        def hook_fn(module, input, output):
            attention_weights.append(output.detach().cpu().numpy())
            
        self.model.attention[0].register_forward_hook(hook_fn)
        
        # Get feature importance from attention weights
        with torch.no_grad():
            X_scaled = self.scaler.transform(self.last_X)
            X_seq = self.prepare_sequences(X_scaled)
            _ = self.model(X_seq)
            
        # Average attention weights across sequences
        avg_attention = np.mean(attention_weights[0], axis=0)
        return avg_attention.flatten()

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