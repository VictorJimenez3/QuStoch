#uses alpha vantage, but that has a limit so we are not using it
#uses finnub which has very lienient usage limits. 
#
import requests
import numpy as np
from datetime import datetime, timedelta
import time

def get_stock_daily_data(api_key, symbol='COF'):
    """
    Fetches daily time series data for a given symbol from Finnhub.
    """
    base_url = 'https://finnhub.io/api/v1/stock/candle'
    
    # Calculate timestamps
    end_timestamp = int(time.time())
    start_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
    
    # Build parameters dictionary
    params = {
        'symbol': symbol,
        'resolution': 'D',
        'from': start_timestamp,
        'to': end_timestamp,
        'token': api_key
    }
    
    # Make request
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('s') == 'no_data':
            print("Error: No data found for the specified symbol and date range.")
            return None
            
    except Exception as e:
        print(f"Error during API request: {e}")
        return None
    
    # Create a dictionary of daily data
    time_series = {}
    for i in range(len(data['t'])):
        date = datetime.fromtimestamp(data['t'][i]).strftime('%Y-%m-%d')
        time_series[date] = {
            "close": data['c'][i],
            "volume": data['v'][i]
        }
    
    # Sort dates in ascending order
    sorted_dates = sorted(time_series.keys())
    return sorted_dates, time_series

def compute_historical_volatility(sorted_dates, time_series, days=30, annualize=True):
    """
    Computes historical volatility based on the standard deviation of logarithmic returns.
    
    Parameters:
      - sorted_dates: List of dates (strings) in ascending order.
      - time_series: Dictionary of daily time series data.
      - days: Number of most recent trading days to include.
      - annualize: If True, annualizes the volatility (using 252 trading days).
      
    Returns:
      - Annualized historical volatility (as a decimal) or None if an error occurs.
    """
    # Use the last 'days' worth of data
    selected_dates = sorted_dates[-days:]
    prices = []
    
    for date in selected_dates:
        try:
            close_price = float(time_series[date]["close"])
            prices.append(close_price)
        except (KeyError, ValueError) as e:
            print(f"Error parsing price data for {date}: {e}")
            return None
    
    # Calculate log returns: ln(P_t / P_(t-1))
    prices = np.array(prices)
    log_returns = np.diff(np.log(prices))
    
    # Compute the standard deviation of the log returns
    vol = np.std(log_returns)
    
    if annualize:
        # Annualize volatility (assume 252 trading days per year)
        vol = vol * np.sqrt(252)
    
    return vol

def main():
    # Replace with your Finnhub API key
    API_KEY = "cujvj7pr01qgs4827llgcujvj7pr01qgs4827lm0"
    SYMBOL = "COF"
    
    # Fetch daily market data
    daily_data = get_stock_daily_data(API_KEY, SYMBOL)
    if daily_data is None:
        return
    
    sorted_dates, time_series = daily_data
    
    # Get the latest day's closing price and volume for display
    latest_date = sorted_dates[-1]
    latest_data = time_series[latest_date]
    
    try:
        market_price = float(latest_data["close"])
        volume = int(latest_data["volume"])
    except (KeyError, ValueError) as e:
        print(f"Error parsing the latest market data: {e}")
        return
    
    print(f"{SYMBOL} closing price on {latest_date}: ${market_price:.2f}")
    print(f"Volume: {volume}")
    
    # Compute historical volatility based on the last 30 trading days
    hist_vol = compute_historical_volatility(sorted_dates, time_series, days=30, annualize=True)
    if hist_vol is not None:
        print(f"Historical Volatility (annualized, based on last 30 trading days): {hist_vol:.2%}")
    else:
        print("Failed to compute historical volatility.")

if __name__ == "__main__":
    main()