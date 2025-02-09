import requests

API_KEY = "cujvj7pr01qgs4827llgcujvj7pr01qgs4827lm0"  # Replace with your key

# Test basic quote endpoint
r = requests.get(f'https://finnhub.io/api/v1/quote?symbol=COF&token={API_KEY}')
print(r.status_code)
print(r.json())