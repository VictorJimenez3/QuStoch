import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define parameters ---
S0 = 100.0     # Current stock price
K = 105.0      # Strike price
T = 30/252     # Time to maturity (30 trading days ≈ 1.5 months)
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility of the underlying asset
trading_hours = 6.5  # Trading hours per day (typical US market)
num_sims = 10_000   # Number of simulated paths
num_steps = int(30 * trading_hours)  # Number of hourly steps (6.5 hours * 30 days)

# --- Step 2: Prepare to store price paths ---
# Create a 2D array where:
#   - Each row represents a time step
#   - Each column represents one simulated path
dt = T / num_steps
S = np.zeros((num_steps + 1, num_sims))
S[0, :] = S0  # All paths start at S0

# --- Step 3: Simulate the paths ---
# For Geometric Brownian Motion (GBM), the stock price evolves as:
#   S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
# where Z ~ N(0, 1).
for t in range(1, num_steps + 1):
    Z = np.random.standard_normal(num_sims)
    S[t, :] = S[t-1, :] * np.exp((r - 0.5 * sigma**2) * dt 
                                 + sigma * np.sqrt(dt) * Z)

# --- Step 4: Calculate the payoff of the European call option ---
# The payoff at maturity for each path is max(S(T) - K, 0).
ST = S[-1, :]                  # Stock price at time T for each path
payoffs = np.maximum(ST - K, 0)

# --- Step 5: Calculate statistics and confidence intervals ---
option_prices = np.exp(-r * T) * payoffs  # Individual option prices
mean_price = np.mean(option_prices)
std_dev = np.std(option_prices)
confidence_level = 0.95
z_score = 1.96  # 95% confidence level
margin_error = z_score * (std_dev / np.sqrt(num_sims))
confidence_interval = (mean_price - margin_error, mean_price + margin_error)

# Calculate Value at Risk (VaR) and Expected Shortfall (ES)
var_95 = np.percentile(option_prices, 5)  # 95% VaR
es_95 = option_prices[option_prices <= var_95].mean()  # 95% ES


# Additional inference statistics
price_range = np.percentile(option_prices, [25, 75])
skewness = np.mean(((option_prices - mean_price) / std_dev) ** 3)
kurtosis = np.mean(((option_prices - mean_price) / std_dev) ** 4) - 3

# Calculate probability of profit
prob_profit = np.mean(option_prices > mean_price)

# Calculate maximum drawdown potential
max_loss = mean_price  # Maximum loss is premium paid
max_gain = np.max(option_prices) - mean_price

print("\n--- Additional Inference Statistics ---")

print(f"Estimated Option Price: {mean_price:.4f}")
print(f"95% Confidence Interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Standard Deviation: {std_dev:.4f}")

print(f"Skewness: {skewness:.4f}")  # Positive means right-tailed
print(f"Kurtosis: {kurtosis:.4f}")  # Higher means more extreme events
print(f"Probability of Profit: {prob_profit:.2%}")
print(f"Risk-Reward Ratio: {max_gain/max_loss:.2f}")


# Calculate Greeks
delta_shift = 0.01 * S0
S_up = S0 + delta_shift
S_down = S0 - delta_shift

# Rerun simulation for shifted prices (simplified for delta calculation)
S_up_final = S_up * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(num_sims))
S_down_final = S_down * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(num_sims))

# Calculate payoffs for shifted prices
payoffs_up = np.maximum(S_up_final - K, 0)
payoffs_down = np.maximum(S_down_final - K, 0)
price_up = np.exp(-r * T) * np.mean(payoffs_up)
price_down = np.exp(-r * T) * np.mean(payoffs_down)

# Greeks calculations
delta = (price_up - price_down) / (2 * delta_shift)
gamma = (price_up - 2 * mean_price + price_down) / (delta_shift ** 2)

# Calculate theta (time decay)
T_next = (T * 252 - 1) / 252  # One day less
if T_next > 0:
    ST_next = S0 * np.exp((r - 0.5 * sigma**2) * T_next + sigma * np.sqrt(T_next) * np.random.standard_normal(num_sims))
    payoffs_next = np.maximum(ST_next - K, 0)
    price_next = np.exp(-r * T_next) * np.mean(payoffs_next)
    theta = (price_next - mean_price)  # Daily theta

# Path-dependent statistics
price_paths = S  # Using already simulated paths
max_drawdowns = np.zeros(num_sims)
strike_crossings = np.zeros(num_sims)
for i in range(num_sims):
    path = price_paths[:, i]
    # Calculate maximum drawdown
    peak = path[0]
    drawdown = 0
    for price in path:
        if price > peak:
            peak = price
        dd = (peak - price) / peak
        if dd > drawdown:
            drawdown = dd
    max_drawdowns[i] = drawdown
    
    # Calculate strike crossings
    crosses = np.sum(np.diff(path > K) != 0)
    strike_crossings[i] = crosses

# Stress testing
stress_scenarios = {
    'high_vol': sigma * 1.5,
    'low_vol': sigma * 0.5,
    'market_crash': S0 * 0.8,
    'market_boom': S0 * 1.2
}

stress_results = {}
for scenario, value in stress_scenarios.items():
    if 'vol' in scenario:
        # Rerun with different volatility
        ST_stress = S0 * np.exp((r - 0.5 * value**2) * T + value * np.sqrt(T) * np.random.standard_normal(num_sims))
    else:
        # Rerun with different starting price
        ST_stress = value * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(num_sims))
    
    payoffs_stress = np.maximum(ST_stress - K, 0)
    stress_results[scenario] = np.exp(-r * T) * np.mean(payoffs_stress)

print("\n--- Greeks ---")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
if T_next > 0:
    print(f"Theta (daily): {theta:.4f}")

print("\n--- Path-Dependent Statistics ---")
print(f"Average Maximum Drawdown: {np.mean(max_drawdowns):.2%}")
print(f"Average Strike Crossings: {np.mean(strike_crossings):.2f}")
print(f"Maximum Strike Crossings: {np.max(strike_crossings):.0f}")

print("\n--- Stress Test Results ---")
for scenario, price in stress_results.items():
    print(f"{scenario}: {price:.4f} (Change: {((price/mean_price - 1)*100):.1f}%)")

# Calculate volatility clustering
returns = np.diff(np.log(S), axis=0)
autocorr = np.mean([np.corrcoef(np.abs(returns[:-1, i]), np.abs(returns[1:, i]))[0,1] 
                    for i in range(num_sims)])

print(f"\nVolatility Clustering (Return Autocorrelation): {autocorr:.4f}")


# --- Step 6: Plot a subset of the simulated paths ---
plt.figure(figsize=(12, 6))
num_paths_to_plot = 10
x_axis = np.linspace(0, 30, num_steps + 1)  # Create x-axis in days
for i in range(num_paths_to_plot):
    plt.plot(x_axis, S[:, i], lw=1)
plt.title("Sample of Monte Carlo Stock Price Paths (Hourly)")
plt.xlabel("Time (Days)")
plt.ylabel("Stock Price")
plt.grid(True)
# Add minor gridlines for hours
plt.grid(True, which='minor', linestyle=':', alpha=0.4)
plt.minorticks_on()
plt.show()