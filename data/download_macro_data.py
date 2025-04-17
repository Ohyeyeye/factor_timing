import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import os

def download_fred_data(start_date: str = '2000-01-01', 
                      end_date: str = '2024-12-31',
                      data_dir: str = 'data'):
    """
    Download macroeconomic data from FRED
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    data_dir : str
        Directory to save the downloaded data
    """
    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # FRED series codes for different interest rates and spreads
    series_codes = {
        # Interest Rates
        'DGS10': '10-Year Treasury Rate',
        'DGS2': '2-Year Treasury Rate',
        'DFF': 'Federal Funds Rate',
        'TB3MS': '3-Month Treasury Bill Rate',
        
        # Credit Spreads
        'BAMLC0A0CM': 'Corporate Bond Spread (Aaa)',
        'BAMLC0A4CBBB': 'Corporate Bond Spread (Baa)',
        'TEDRATE': 'TED Spread',
        'BAMLC0A1CAAAEY': 'Corporate Bond Spread (Aa)',
        
        # Yield Curve
        'T10Y2Y': '10-Year Treasury - 2-Year Treasury Spread',
        'T10Y3M': '10-Year Treasury - 3-Month Treasury Spread'
    }
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download each series
    for code, description in series_codes.items():
        try:
            print(f"Downloading {description} ({code})...")
            df = web.DataReader(code, 'fred', start, end)
            
            # Save to CSV
            filename = f"{code.lower()}.csv"
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath)
            print(f"Saved {description} to {filepath}")
            
        except Exception as e:
            print(f"Error downloading {code}: {str(e)}")
    
    # Download and save all series in one file for convenience
    try:
        print("\nDownloading all series into one file...")
        all_data = web.DataReader(list(series_codes.keys()), 'fred', start, end)
        all_data.to_csv(os.path.join(data_dir, 'all_macro_data.csv'))
        print("Saved all series to all_macro_data.csv")
    except Exception as e:
        print(f"Error downloading combined data: {str(e)}")

def main():
    # Example usage
    download_fred_data(
        start_date='2000-01-01',
        end_date='2024-12-31',
        data_dir='data'
    )

if __name__ == "__main__":
    main() 