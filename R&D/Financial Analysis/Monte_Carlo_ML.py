import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Step 1: Define parameters ---
S0 = 100.0     # Current stock price
K = 105.0      # Strike price
T = 30/252     # Time to maturity (30 trading days ≈ 1.5 months)
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility of the underlying asset
trading_hours = 6.5  # Trading hours per day (typical US market)
num_sims = 10_000   # Number of simulated paths
num_steps = int(30 * trading_hours)  # Number of hourly steps (6.5 hours * 30 days)
S = np.zeros((num_steps + 1, num_sims))

def run_monte_carlo_iteration(S0, K, T, r, sigma, num_sims=10_000, num_steps=195):
    """Run a single Monte Carlo simulation and return key metrics"""
    dt = T / num_steps
    S[0, :] = S0 
    
    # Simulate paths
    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_sims)
        S[t, :] = S[t-1, :] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Calculate metrics
    ST = S[-1, :]
    payoffs = np.maximum(ST - K, 0)
    option_prices = np.exp(-r * T) * payoffs
    
    # Calculate various statistics
    mean_price = np.mean(option_prices)
    std_dev = np.std(option_prices)
    skewness = np.mean(((option_prices - mean_price) / std_dev) ** 3)
    kurtosis = np.mean(((option_prices - mean_price) / std_dev) ** 4) - 3
    var_95 = np.percentile(option_prices, 5)
    es_95 = option_prices[option_prices <= var_95].mean()
    
    return {
        'mean_price': mean_price,
        'std_dev': std_dev,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'es_95': es_95,
        'max_price': np.max(option_prices),
        'min_price': np.min(option_prices),
        'input_S0': S0,
        'input_K': K,
        'input_T': T,
        'input_r': r,
        'input_sigma': sigma
    }

def generate_training_data(num_iterations=1000):
    """Generate multiple iterations with varying input parameters"""
    results = []
    
    # Generate variations of input parameters
    for _ in tqdm(range(num_iterations), desc="Generating training data"):
        # Randomly vary input parameters within reasonable ranges
        S0 = np.random.uniform(80, 120)
        K = np.random.uniform(S0 * 0.8, S0 * 1.2)
        T = np.random.uniform(10/252, 60/252)
        r = np.random.uniform(0.01, 0.08)
        sigma = np.random.uniform(0.1, 0.4)
        
        result = run_monte_carlo_iteration(S0, K, T, r, sigma)
        results.append(result)
    
    return pd.DataFrame(results)

def train_ml_models(df):
    """Train machine learning models on the simulation results"""
    # Prepare features and targets
    features = ['input_S0', 'input_K', 'input_T', 'input_r', 'input_sigma']
    targets = ['mean_price', 'std_dev', 'var_95', 'es_95']
    
    models = {}
    scores = {}
    
    for target in targets:
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        models[target] = model
        scores[target] = score
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature importance for {target}:")
        print(importance)
        
    return models, scores

def analyze_patterns(df):
    """Analyze patterns and correlations in the results"""
    # Correlation analysis
    correlation_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Monte Carlo Simulation Results')
    plt.tight_layout()
    plt.show()
    
    # Distribution analysis
    metrics = ['mean_price', 'std_dev', 'var_95', 'es_95']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        sns.histplot(df[metric], ax=ax)
        ax.set_title(f'Distribution of {metric}')
    plt.tight_layout()
    plt.show()

def main():
    # Generate training data
    print("Generating training data...")
    df = generate_training_data(num_iterations=1000)
    
    # Train ML models
    print("\nTraining ML models...")
    models, scores = train_ml_models(df)
    
    # Print model performance
    print("\nModel Performance (R² scores):")
    for target, score in scores.items():
        print(f"{target}: {score:.4f}")
    
    # Analyze patterns
    print("\nAnalyzing patterns...")
    analyze_patterns(df)
    
    # Example prediction
    example_input = pd.DataFrame({
        'input_S0': [100],
        'input_K': [105],
        'input_T': [30/252],
        'input_r': [0.05],
        'input_sigma': [0.2]
    })
    
    print("\nExample prediction for standard inputs:")
    for target, model in models.items():
        prediction = model.predict(example_input)[0]
        print(f"{target}: {prediction:.4f}")

if __name__ == "__main__":
    main()

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