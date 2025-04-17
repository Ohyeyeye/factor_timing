import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class BaseRegimeClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare and scale features"""
        return self.scaler.fit_transform(X)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model"""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime"""
        raise NotImplementedError

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

class LSTMRegimeClassifier(BaseRegimeClassifier):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = LSTMModel(input_size, hidden_size, num_layers, 1)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train LSTM model"""
        X_scaled = self.prepare_features(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)  # Add sequence dimension
        y_tensor = torch.FloatTensor(y.values)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime using trained LSTM"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs.squeeze().numpy() > 0.5).astype(int)
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
    def __init__(self, n_components: int = 2):
        super().__init__()
        from hmmlearn import hmm
        self.model = hmm.GaussianHMM(n_components=n_components)
        
    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Train HMM model"""
        X_scaled = self.prepare_features(X)
        self.model.fit(X_scaled)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime using trained HMM"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled) 