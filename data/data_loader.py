import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging

class DataLoader:
    def __init__(self, data_dir: str = None):
        """Initialize data loader"""
        self.data_dir = data_dir or 'data'
        self.logger = logging.getLogger(__name__)
        self.train_start_date = None
        self.train_end_date = None
        self.test_start_date = None
        self.test_end_date = None
        self.ff_factors = None
        self.macro_data = None
        self.market_data = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        # Validate data directory
        if not os.path.exists(self.data_dir):
            self.logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            os.makedirs(self.data_dir)
    
    def set_date_ranges(self, train_start: str, train_end: str, test_start: str, test_end: str):
        """Set the date ranges for training and testing"""
        self.train_start_date = train_start
        self.train_end_date = train_end
        self.test_start_date = test_start
        self.test_end_date = test_end
    
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
            # Read CSV with first column as index
            self.ff_factors = pd.read_csv(ff_path, index_col=0)
            
            # Convert index to datetime (assuming format YYYYMM)
            self.ff_factors.index = pd.to_datetime(self.ff_factors.index.astype(str).str.zfill(6), format='%Y%m')
            
            # Add date column
            self.ff_factors['date'] = self.ff_factors.index
            
        except Exception as e:
            raise ValueError(f"Error reading Fama-French factors file: {str(e)}")
        
        # Validate and process data
        required_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        self._validate_dataframe(self.ff_factors, required_columns, "Fama-French factors")
        
        # Filter by date range
        self.ff_factors = self.ff_factors.loc[start:end]
        
        if self.ff_factors.empty:
            raise ValueError(f"No Fama-French factor data available for the specified date range: {start_date} to {end_date}")
            
        return self.ff_factors
    
    def load_macro_data(self,
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        Load macroeconomic data from CSV files or download if not available
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Combined macroeconomic data
        """
        start, end = self._validate_date_range(start_date, end_date)
        
        # Check if combined macro data file exists
        combined_macro_path = os.path.join(self.data_dir, 'combined_macro_data.csv')
        if os.path.exists(combined_macro_path):
            self.logger.info("Loading existing macro data...")
            try:
                macro_data = pd.read_csv(combined_macro_path, index_col=0, parse_dates=True)
                macro_data = macro_data.loc[start:end]
                if not macro_data.empty:
                    self.logger.info(f"Loaded existing macro data with {len(macro_data)} rows")
                    return macro_data
            except Exception as e:
                self.logger.warning(f"Error loading existing macro data: {str(e)}")
        
        # Required macro data files
        required_files = ['cpi.csv', 'GDP.csv', 'UNRATE.csv']
        for file in required_files:
            if not os.path.exists(os.path.join(self.data_dir, file)):
                raise FileNotFoundError(f"Required macro data file not found: {file}")
        
        try:
            # Load CPI data
            cpi = pd.read_csv(os.path.join(self.data_dir, 'cpi.csv'))
            cpi['date'] = pd.to_datetime(cpi['observation_date'])
            cpi.set_index('date', inplace=True)
            cpi = cpi.rename(columns={'CORESTICKM159SFRBATL': 'CPI'})
            
            # Load GDP data
            gdp = pd.read_csv(os.path.join(self.data_dir, 'GDP.csv'))
            gdp['date'] = pd.to_datetime(gdp['observation_date'])
            gdp.set_index('date', inplace=True)
            gdp = gdp.rename(columns={'GDP': 'GDP'})
            
            # Load unemployment rate data
            unrate = pd.read_csv(os.path.join(self.data_dir, 'UNRATE.csv'))
            unrate['date'] = pd.to_datetime(unrate['observation_date'])
            unrate.set_index('date', inplace=True)
            unrate = unrate.rename(columns={'UNRATE': 'UNRATE'})
            
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
                credit_spread['date'] = pd.to_datetime(credit_spread['observation_date'])
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
                treasury['date'] = pd.to_datetime(treasury['observation_date'])
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
        
        # Forward fill missing values using newer pandas syntax
        self.macro_data = self.macro_data.ffill()
        
        # Save combined macro data
        try:
            self.macro_data.to_csv(combined_macro_path)
            self.logger.info(f"Saved combined macro data to {combined_macro_path}")
        except Exception as e:
            self.logger.warning(f"Could not save combined macro data: {str(e)}")
        
        # Log data summary
        self.logger.info(f"Loaded macro data with {len(self.macro_data)} rows and {len(self.macro_data.columns)} columns")
        self.logger.info(f"Date range: {self.macro_data.index.min()} to {self.macro_data.index.max()}")
        
        return self.macro_data
    
    def load_market_data(self,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """
        Load market data (S&P 500, VIX, etc.) from saved file or download if not available
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Market data with returns and volatility
        """
        start, end = self._validate_date_range(start_date, end_date)
        
        # Check if market data file exists
        market_data_path = os.path.join(self.data_dir, f'market_data_{start_date}_{end_date}.csv')
        if os.path.exists(market_data_path):
            self.logger.info("Loading existing market data...")
            try:
                market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
                market_data = market_data.loc[start:end]
                if not market_data.empty:
                    self.logger.info(f"Loaded existing market data with {len(market_data)} rows")
                    return market_data
            except Exception as e:
                self.logger.warning(f"Error loading existing market data: {str(e)}")
        
        try:
            # Download S&P 500 data
            sp500 = yf.download('^GSPC', start=start, end=end)
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500.columns = sp500.columns.get_level_values(0)
            
            vix = yf.download('^VIX', start=start, end=end)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            
            if sp500.empty or vix.empty:
                raise ValueError("Failed to download market data from Yahoo Finance")
            
            # Use Close price if Adj Close is not available
            sp500_price = sp500['Close'] if 'Adj Close' not in sp500.columns else sp500['Adj Close']
            vix_price = vix['Close'] if 'Adj Close' not in vix.columns else vix['Adj Close']
            
            # Calculate returns and realized volatility
            sp500_returns = sp500_price.pct_change()
            realized_vol = sp500_returns.rolling(window=21).std() * np.sqrt(252)
            
            # Create market data DataFrame
            self.market_data = pd.DataFrame({
                'SP500_Return': sp500_returns,
                'VIX': vix_price,
                'Realized_Vol': realized_vol
            })
            
            # Handle missing values more robustly
            self.market_data = (self.market_data
                              .ffill(limit=5)  # Forward fill with limit
                              .bfill(limit=5)  # Backward fill with limit
                              .fillna(0))  # Fill any remaining NaNs with 0
            
            # Validate market data
            self._validate_dataframe(self.market_data, 
                                  ['SP500_Return', 'VIX', 'Realized_Vol'],
                                  "Market data")
            
            # Save market data
            try:
                self.market_data.to_csv(market_data_path)
                self.logger.info(f"Saved market data to {market_data_path}")
            except Exception as e:
                self.logger.warning(f"Could not save market data: {str(e)}")
            
            # Log data summary
            self.logger.info(f"Loaded market data with {len(self.market_data)} rows")
            self.logger.info(f"Date range: {self.market_data.index.min()} to {self.market_data.index.max()}")
            
            return self.market_data
            
        except Exception as e:
            raise ValueError(f"Error downloading market data: {str(e)}")
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for regime classification"""
        if not all([self.train_start_date, self.train_end_date]):
            raise ValueError("Training date ranges not set. Call set_date_ranges first.")
            
        # Load macro and market data
        macro_data = self.load_macro_data(self.train_start_date, self.train_end_date)
        market_data = self.load_market_data(self.train_start_date, self.train_end_date)
        
        # Combine features
        X = pd.concat([macro_data, market_data], axis=1)
        
        # Create target variable based on market regime
        # Use VIX as a proxy for market regime
        vix = market_data['VIX']
        vix_returns = vix.pct_change()
        
        # Define regimes based on VIX returns
        # 0: Low volatility regime (VIX returns below median)
        # 1: High volatility regime (VIX returns above median)
        y = (vix_returns > vix_returns.median()).astype(int)
        
        # Drop first row due to NaN in returns
        X = X.iloc[1:]
        y = y.iloc[1:]
        
        # Ensure consistent lengths
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Log data preparation info
        self.logger.info(f"Prepared training data with {len(X)} samples")
        self.logger.info(f"Class distribution: {dict(y.value_counts())}")
        
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