import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime

def get_stock_daily_data(symbol):
    """
    Fetches daily historical data for a given symbol using yfinance.
    Retrieves approximately one year of data.
    
    Returns:
      - A DataFrame with date-indexed daily data containing columns like Close and Volume.
    """
    ticker = yf.Ticker(symbol)
    # Download ~1 year of historical data with daily resolution.
    df = ticker.history(period="1y", interval="1d")
    
    if df.empty:
        print(f"Error: No data found for {symbol} in the specified date range.")
        return None
    
    # Format the index to a string date format (YYYY-MM-DD)
    df.index = df.index.strftime('%Y-%m-%d')
    return df

def compute_historical_volatility(df, days=30, annualize=True):
    """
    Computes historical volatility based on the standard deviation of logarithmic returns.
    
    Parameters:
      - df: DataFrame containing at least the 'Close' column.
      - days: Number of most recent trading days to include.
      - annualize: If True, annualizes the volatility (assuming 252 trading days per year).
      
    Returns:
      - Annualized historical volatility (as a decimal) or None if there is an error.
    """
    # Use the last 'days' of closing prices.
    close_prices = df['Close'][-days:]
    
    if len(close_prices) < days:
        print("Error: Not enough data to compute volatility.")
        return None

    # Calculate log returns: ln(P_t / P_(t-1))
    log_returns = np.diff(np.log(close_prices))
    vol = np.std(log_returns)
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol

def compute_beta(stock_df, market_symbol="^GSPC", days=30): 
    """
    Computes the beta of a stock relative to a market index.
    
    Beta > 1: The stock is more volatile than the market. 
    
    Beta < 1: The stock is less volatile than the market.
    Parameters:
      - stock_df: DataFrame of the stock's historical data.
      - market_symbol: The ticker symbol for the market index (default is '^GSPC' for the S&P 500).
      - days: Number of recent trading days to include in the calculation.
      
    Returns:
      - Beta value or None if the calculation fails.
    """
    # Fetch market data using the same helper function.
    market_df = get_stock_daily_data(market_symbol)
    if market_df is None:
        print(f"Error: No market data found for {market_symbol}.")
        return None
    
    # Merge the stock and market data on the date index.
    df_combined = pd.merge(
        stock_df[['Close']], market_df[['Close']],
        left_index=True, right_index=True,
        suffixes=('_stock', '_market')
    )
    
    # Sort by index to ensure chronological order.
    df_combined = df_combined.sort_index()
    
    # Ensure there are enough data points; note we need one extra point to compute returns.
    if len(df_combined) < days + 1:
        print("Error: Not enough overlapping data to compute beta.")
        return None
    
    # Use the most recent (days+1) data points.
    df_combined = df_combined.tail(days + 1)
    
    # Compute daily log returns for both stock and market.
    stock_returns = np.diff(np.log(df_combined['Close_stock'].values))
    market_returns = np.diff(np.log(df_combined['Close_market'].values))
    
    # Calculate covariance matrix and derive beta.
    cov_matrix = np.cov(stock_returns, market_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    return beta

def get_stock_info(symbol):
    """
    Retrieves stock data for the specified symbol:
      - Downloads approximately one year of daily data using yfinance.
      - Extracts the latest day's closing price and volume.
      - Computes the annualized historical volatility based on the last 30 trading days.
      - Computes the beta (systematic risk) relative to the S&P 500.
      
    Prints the data and returns a dictionary containing:
        'latest_date', 'latest_price', 'latest_volume', 'historical_volatility', and 'beta'
        
    Returns:
        A dictionary with the stock information, or None if data retrieval fails.
    """
    df = get_stock_daily_data(symbol)
    if df is None:
        return None

    # Get the latest day's data for display.
    latest_date = df.index[-1]
    latest_price = df.loc[latest_date, 'Close']
    latest_volume = df.loc[latest_date, 'Volume']
    
    print(f"{symbol} closing price on {latest_date}: ${latest_price:.2f}")
    print(f"Volume: {latest_volume}")
    
    # Compute historical volatility based on the last 30 trading days.
    hist_vol = compute_historical_volatility(df, days=30, annualize=True)
    if hist_vol is not None:
        print(f"Historical Volatility (annualized, based on last 30 trading days): {hist_vol:.2%}")
    else:
        print("Failed to compute historical volatility.")
    
    # Compute beta relative to the S&P 500.
    beta = compute_beta(df, market_symbol="^GSPC", days=30)
    if beta is not None:
        print(f"Beta (based on last 30 trading days): {beta:.2f}")
    else:
        print("Failed to compute beta.")
    
    return {
        "latest_date": latest_date,
        "latest_price": latest_price,
        "latest_volume": latest_volume,
        "historical_volatility": hist_vol,
        "beta": beta
    }

# Example usage:
if __name__ == "__main__":
    symbol = "AAPL"
    symbol1 = "COF"

    stock_info = get_stock_info(symbol)
    stock_info1 = get_stock_info(symbol1)
    
    if stock_info is not None:
        print("\nReturned Stock Information for", symbol)
        print(stock_info)
    
    if stock_info1 is not None:
        print("\nReturned Stock Information for", symbol1)
        print(stock_info1)
