import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging

class DataLoader:
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the DataLoader with a data directory.
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.ff_factors = None
        self.macro_data = None
        self.market_data = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate data directory
        if not os.path.exists(data_dir):
            self.logger.warning(f"Data directory {data_dir} does not exist. Creating it.")
            os.makedirs(data_dir)
    
    def _validate_date_range(self, start_date: str, end_date: str) -> Tuple[datetime, datetime]:
        """
        Validate and convert date strings to datetime objects.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Tuple[datetime, datetime]: Validated start and end dates
            
        Raises:
            ValueError: If dates are invalid or end date is before start date
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format. Error: {str(e)}")
        
        if end < start:
            raise ValueError("End date must be after start date")
            
        return start, end
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str], 
                          data_type: str) -> None:
        """
        Validate DataFrame structure and content.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (List[str]): List of required column names
            data_type (str): Type of data being validated
            
        Raises:
            ValueError: If DataFrame is invalid
        """
        if df is None or df.empty:
            raise ValueError(f"{data_type} data is empty")
            
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {data_type} data: {missing_cols}")
            
        if df.isnull().any().any():
            self.logger.warning(f"Found missing values in {data_type} data")
            
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{data_type} data index must be datetime")
    
    def load_fama_french_factors(self, 
                               start_date: str,
                               end_date: str) -> pd.DataFrame:
        """
        Load Fama-French 5-factor data from ff.csv
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Fama-French factor returns
            
        Raises:
            FileNotFoundError: If ff.csv is not found
            ValueError: If data is invalid
        """
        start, end = self._validate_date_range(start_date, end_date)
        
        # Load Fama-French factors
        ff_path = os.path.join(self.data_dir, 'ff.csv')
        if not os.path.exists(ff_path):
            raise FileNotFoundError(f"Fama-French factors file not found at {ff_path}")
            
        try:
            self.ff_factors = pd.read_csv(ff_path)
        except Exception as e:
            raise ValueError(f"Error reading Fama-French factors file: {str(e)}")
        
        # Validate and process data
        required_columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        self._validate_dataframe(self.ff_factors, required_columns, "Fama-French factors")
        
        # Convert date column to datetime
        self.ff_factors['date'] = pd.to_datetime(self.ff_factors['date'])
        self.ff_factors.set_index('date', inplace=True)
        
        # Filter by date range
        self.ff_factors = self.ff_factors.loc[start:end]
        
        if self.ff_factors.empty:
            raise ValueError(f"No Fama-French factor data available for the specified date range: {start_date} to {end_date}")
            
        return self.ff_factors
    
    def load_macro_data(self,
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        Load macroeconomic data from CSV files
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Combined macroeconomic data
            
        Raises:
            FileNotFoundError: If required macro data files are not found
            ValueError: If data is invalid
        """
        start, end = self._validate_date_range(start_date, end_date)
        
        # Required macro data files
        required_files = ['cpi.csv', 'GDP.csv', 'UNRATE.csv']
        for file in required_files:
            if not os.path.exists(os.path.join(self.data_dir, file)):
                raise FileNotFoundError(f"Required macro data file not found: {file}")
        
        try:
            # Load CPI data
            cpi = pd.read_csv(os.path.join(self.data_dir, 'cpi.csv'))
            cpi['date'] = pd.to_datetime(cpi['date'])
            cpi.set_index('date', inplace=True)
            
            # Load GDP data
            gdp = pd.read_csv(os.path.join(self.data_dir, 'GDP.csv'))
            gdp['date'] = pd.to_datetime(gdp['date'])
            gdp.set_index('date', inplace=True)
            
            # Load unemployment rate data
            unrate = pd.read_csv(os.path.join(self.data_dir, 'UNRATE.csv'))
            unrate['date'] = pd.to_datetime(unrate['date'])
            unrate.set_index('date', inplace=True)
            
        except Exception as e:
            raise ValueError(f"Error reading macro data files: {str(e)}")
        
        # Validate required data
        self._validate_dataframe(cpi, ['CPI'], "CPI data")
        self._validate_dataframe(gdp, ['GDP'], "GDP data")
        self._validate_dataframe(unrate, ['UNRATE'], "Unemployment data")
        
        # Load US corporate credit spread
        try:
            credit_spread_path = os.path.join(self.data_dir, 'bamlc0a4cbbb.csv')
            if os.path.exists(credit_spread_path):
                credit_spread = pd.read_csv(credit_spread_path)
                credit_spread['date'] = pd.to_datetime(credit_spread['date'])
                credit_spread.set_index('date', inplace=True)
                credit_spread = credit_spread.rename(columns={credit_spread.columns[0]: 'CORP_SPREAD'})
            else:
                self.logger.warning("US corporate credit spread file not found")
                credit_spread = None
        except Exception as e:
            self.logger.warning(f"Could not load US corporate credit spread: {str(e)}")
            credit_spread = None
        
        # Load 10-year Treasury rate
        try:
            treasury_path = os.path.join(self.data_dir, 'dgs10.csv')
            if os.path.exists(treasury_path):
                treasury = pd.read_csv(treasury_path)
                treasury['date'] = pd.to_datetime(treasury['date'])
                treasury.set_index('date', inplace=True)
                treasury = treasury.rename(columns={treasury.columns[0]: 'TREASURY_10Y'})
            else:
                self.logger.warning("10-year Treasury rate file not found")
                treasury = None
        except Exception as e:
            self.logger.warning(f"Could not load 10-year Treasury rate: {str(e)}")
            treasury = None
        
        # Combine all macro data
        macro_data_frames = [
            gdp['GDP'],
            cpi['CPI'],
            unrate['UNRATE']
        ]
        
        # Add credit spread if available
        if credit_spread is not None:
            macro_data_frames.append(credit_spread['CORP_SPREAD'])
            
        # Add Treasury rate if available
        if treasury is not None:
            macro_data_frames.append(treasury['TREASURY_10Y'])
        
        # Combine all data
        self.macro_data = pd.concat(macro_data_frames, axis=1)
        
        # Filter by date range
        self.macro_data = self.macro_data.loc[start:end]
        
        if self.macro_data.empty:
            raise ValueError(f"No macro data available for the specified date range: {start_date} to {end_date}")
        
        # Forward fill missing values
        self.macro_data = self.macro_data.fillna(method='ffill')
        
        # Log data summary
        self.logger.info(f"Loaded macro data with {len(self.macro_data)} rows and {len(self.macro_data.columns)} columns")
        self.logger.info(f"Date range: {self.macro_data.index.min()} to {self.macro_data.index.max()}")
        
        return self.macro_data
    
    def load_market_data(self,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """
        Load market data (S&P 500, VIX, etc.)
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Market data with returns and volatility
            
        Raises:
            ValueError: If data download fails or is invalid
        """
        start, end = self._validate_date_range(start_date, end_date)
        
        try:
            # Download S&P 500 data
            sp500 = yf.download('^GSPC', start=start, end=end)
            vix = yf.download('^VIX', start=start, end=end)
            
            if sp500.empty or vix.empty:
                raise ValueError("Failed to download market data from Yahoo Finance")
            
            # Calculate returns and realized volatility
            sp500_returns = sp500['Adj Close'].pct_change()
            realized_vol = sp500_returns.rolling(window=21).std() * np.sqrt(252)
            
            self.market_data = pd.DataFrame({
                'SP500_Return': sp500_returns,
                'VIX': vix['Adj Close'],
                'Realized_Vol': realized_vol
            })
            
            # Forward fill missing values
            self.market_data = self.market_data.fillna(method='ffill')
            
            # Validate market data
            self._validate_dataframe(self.market_data, 
                                  ['SP500_Return', 'VIX', 'Realized_Vol'],
                                  "Market data")
            
            # Log data summary
            self.logger.info(f"Loaded market data with {len(self.market_data)} rows")
            self.logger.info(f"Date range: {self.market_data.index.min()} to {self.market_data.index.max()}")
            
            return self.market_data
            
        except Exception as e:
            raise ValueError(f"Error downloading market data: {str(e)}")
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare combined dataset for regime classification
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
            
        Raises:
            ValueError: If required data is not loaded or invalid
        """
        if self.macro_data is None or self.market_data is None:
            raise ValueError("Please load macro and market data first")
            
        # Combine macro and market data
        X = pd.concat([self.macro_data, self.market_data], axis=1)
        
        # Create regime labels based on VIX and SP500 returns
        # High vol regime: VIX > 20
        # Low vol regime: VIX <= 20
        y = (self.market_data['VIX'] > 20).astype(int)
        
        # Drop any remaining NaN values
        X = X.dropna()
        y = y[X.index]  # Align y with X after dropping NaNs
        
        if X.empty:
            raise ValueError("No valid data available after preprocessing")
            
        # Log data summary
        self.logger.info(f"Prepared training data with {len(X)} samples")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def get_factor_returns(self) -> pd.DataFrame:
        """Get the loaded Fama-French factor returns"""
        if self.ff_factors is None:
            raise ValueError("Please load Fama-French factors first")
        return self.ff_factors
    
    def get_macro_data(self) -> pd.DataFrame:
        """Get the loaded macroeconomic data"""
        if self.macro_data is None:
            raise ValueError("Please load macro data first")
        return self.macro_data
    
    def get_market_data(self) -> pd.DataFrame:
        """Get the loaded market data"""
        if self.market_data is None:
            raise ValueError("Please load market data first")
        return self.market_data 