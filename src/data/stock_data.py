"""Stock data collection module using Yahoo Finance API"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional


class StockDataCollector:
    """Collects and processes stock market data"""
    
    # Magnificent Seven stocks
    MAGNIFICENT_SEVEN = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
    
    def __init__(self, tickers: Optional[List[str]] = None):
        """
        Initialize the data collector
        
        Args:
            tickers: List of stock tickers. Defaults to Magnificent Seven.
        """
        self.tickers = tickers if tickers else self.MAGNIFICENT_SEVEN
        self.data = None
        self.returns = None
        
    def fetch_data(self, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None,
                   period: str = '1y') -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Period to fetch if dates not specified (e.g., '1y', '2y', '5y')
            
        Returns:
            DataFrame with adjusted closing prices
        """
        print(f"Fetching data for: {', '.join(self.tickers)}")
        
        if start_date and end_date:
            data = yf.download(self.tickers, start=start_date, end=end_date, 
                             progress=False)['Adj Close']
        else:
            data = yf.download(self.tickers, period=period, progress=False)['Adj Close']
        
        # Handle single ticker case
        if len(self.tickers) == 1:
            data = data.to_frame()
            data.columns = self.tickers
            
        self.data = data
        print(f"Fetched {len(data)} days of data")
        return data
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns
        
        Returns:
            DataFrame with daily returns
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
            
        self.returns = self.data.pct_change().dropna()
        return self.returns
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate summary statistics for the stocks
        
        Returns:
            DataFrame with summary statistics
        """
        if self.returns is None:
            self.calculate_returns()
            
        stats = pd.DataFrame({
            'Mean Return': self.returns.mean(),
            'Std Dev': self.returns.std(),
            'Min': self.returns.min(),
            'Max': self.returns.max(),
            'Sharpe Ratio (Annualized)': (self.returns.mean() * 252) / (self.returns.std() * (252 ** 0.5))
        })
        
        return stats
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between stocks
        
        Returns:
            Correlation matrix
        """
        if self.returns is None:
            self.calculate_returns()
            
        return self.returns.corr()
