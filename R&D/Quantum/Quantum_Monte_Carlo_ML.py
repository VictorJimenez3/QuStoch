import pennylane as qml
import numpy as np
import scipy
import scipy.stats
from time import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import seaborn as sns
from tqdm import tqdm  # Add this import at the top with other imports


# --- Step 1: Define parameters ---
S0 = 100.0     # Current stock price
K = 105.0      # Strike price
T = 30/252  # time (e.g. in years)
r = 0.05  # risk free rate
sigma = 0.2  # volatility
trading_hours = 6.5  # Trading hours per day (typical US market)
num_steps = int(30 * trading_hours)  # Number of hourly steps (6.5 hours * 30 days)
dt = T / num_steps # Time step

variance = 1 # Variance of the Normal Distribution being used.
cutoff_factor = 4


def payoff_func(spot, strike, rate, volatility, time, x):
    payoffs = []
    all_paths = []

    for W in x:
        price = [spot]
        price.append(price[-1] * np.exp(volatility * W + (rate - 0.5 * volatility ** 2) * dt))
        price_array = np.array(price)
        all_paths.append(price_array)
        payoff = call_payoffs(np.expand_dims(price_array[1:], axis=0), strike, spot)[0]
        payoffs.append(payoff)

    if np.max(payoffs) == 0:
        payoffs[0] = 1e-10
    return np.array(payoffs)*np.exp(- rate * time), np.array(all_paths)



def call_payoffs(paths, strike, spot):
    spots = np.full((paths.shape[0], 1), spot)
    paths = np.append(spots, paths, axis=1)
    
    means = scipy.stats.mstats.gmean(paths, axis=1)
    
    asian_payoffs = means - strike
    asian_payoffs[asian_payoffs < 0] = 0

    return asian_payoffs



def normal_dist(n, variance, cutoff_factor):
    dim = 2 ** n
    points = np.random.normal(0, np.sqrt(variance), dim)
    points = np.clip(points, -cutoff_factor * np.sqrt(variance), cutoff_factor * np.sqrt(variance))
    prob = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(- 0.5 * points ** 2 / variance)
    prob_renorm = prob / np.sum(prob)
    return [points, prob_renorm]




def Quantum_Monte_Carlo(spot=100, strike=100, rate=0.05, volatility=0.2, time=1, variance=1.0, cutoff_factor=4, n_disc=4, n_pe=10):


    N_disc, N_pe = 2**n_disc, 2**n_pe
    
    x, p = normal_dist(n_disc, T, cutoff_factor)
    asian_payoff, paths = payoff_func(spot, strike, rate, volatility, time, x)
    
    # Calculate additional statistics
    mean_price = np.mean(asian_payoff)
    std_dev = np.std(asian_payoff)
    
    # Calculate confidence interval
    z_score = 1.96  # 95% confidence level
    margin_error = z_score * (std_dev / np.sqrt(len(asian_payoff)))
    confidence_interval = (mean_price - margin_error, mean_price + margin_error)
    
    # Calculate higher moments
    skewness = np.mean(((asian_payoff - mean_price) / std_dev) ** 3)
    kurtosis = np.mean(((asian_payoff - mean_price) / std_dev) ** 4) - 3
    
    # Calculate probability of profit and risk-reward metrics
    prob_profit = np.mean(asian_payoff > mean_price)
    max_loss = mean_price  # Maximum loss is premium paid
    max_gain = np.max(asian_payoff) - mean_price
    
    # Calculate path-dependent statistics
    max_drawdowns = []
    strike_crossings = []
    
    for path in paths:
        # Maximum drawdown
        peak = path[0]
        max_drawdown = 0
        crossings = 0
        prev_above_strike = path[0] > strike
        
        for price in path[1:]:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
            # Count strike crossings
            curr_above_strike = price > strike
            if curr_above_strike != prev_above_strike:
                crossings += 1
            prev_above_strike = curr_above_strike
            
        max_drawdowns.append(max_drawdown)
        strike_crossings.append(crossings)
    
    # Stress test scenarios with expanded scenarios and detailed reporting
    stress_scenarios = {
        'High Volatility': {'volatility': volatility * 1.5},
        'Low Volatility': {'volatility': volatility * 0.5},
        'Market Crash': {'spot': spot * 0.8, 'volatility': volatility * 1.3},
        'Market Boom': {'spot': spot * 1.2},
        'Low Interest Rate': {'rate': rate * 0.5},
        'High Interest Rate': {'rate': rate * 1.5}
    }
    
    stress_results = {}
    for scenario, params in stress_scenarios.items():
        # Set default parameters
        test_spot = params.get('spot', spot)
        test_vol = params.get('volatility', volatility)
        test_rate = params.get('rate', rate)
        
        # Calculate price under stress scenario
        stress_price = payoff_func(test_spot, strike, test_rate, test_vol, time, x)[0].mean()
        stress_results[scenario] = stress_price
    
    
    # Calculate volatility clustering
    returns = np.array([np.diff(np.log(path)) for path in paths])
    
    # Add error handling for autocorrelation calculation
    try:
        autocorr = np.mean([
            np.corrcoef(np.abs(returns[i][:-1]), np.abs(returns[i][1:]))[0,1] 
            for i in range(len(returns))
            if len(returns[i]) > 1  # Check if we have enough data points
        ])
        if np.isnan(autocorr):
            autocorr = 0.0
    except:
        autocorr = 0.0
    
    # Ensure we don't divide by zero in risk-reward calculation
    if max_loss != 0:
        risk_reward_ratio = max_gain/max_loss
    else:
        risk_reward_ratio = float('inf')
        
    # Calculate Greeks
    delta_shift = 0.01 * spot
    gamma_shift = 0.01 * spot
    theta_shift = 1/252  # One day shift
    
    # Delta calculation (dV/dS)
    price_up = payoff_func(spot + delta_shift, strike, rate, volatility, time, x)[0].mean()
    price_down = payoff_func(spot - delta_shift, strike, rate, volatility, time, x)[0].mean()
    delta = (price_up - price_down) / (2 * delta_shift)
    
    # Gamma calculation (d²V/dS²)
    gamma = (price_up - 2*mean_price + price_down) / (delta_shift**2)
    
    # Theta calculation (dV/dt)
    if time > theta_shift:
        price_next = payoff_func(spot, strike, rate, volatility, time - theta_shift, x)[0].mean()
        theta = (price_next - mean_price) / theta_shift
    else:
        theta = 0
    
    # Instead of printing, store all metrics in a dictionary
    metrics = {
        "Option Price": {
            "Estimated Price": mean_price,
            "95% Confidence Interval": confidence_interval if not np.isnan(std_dev) else None,
            "Standard Deviation": std_dev if not np.isnan(std_dev) else None,
            "Skewness": skewness if not np.isnan(std_dev) else None,
            "Kurtosis": kurtosis if not np.isnan(std_dev) else None,
            "Probability of Profit": prob_profit,
            "Risk-Reward Ratio": risk_reward_ratio
        },
        "Greeks": {
            "Delta": delta,
            "Gamma": gamma,
            "Theta (daily)": theta if time > theta_shift else None
        },
        "Stress Test Results": {
            scenario: {
                "Price": price,
                "Change": ((price/mean_price - 1) * 100)
            } for scenario, price in stress_results.items()
        },
        "Path-Dependent Statistics": {
            "Average Maximum Drawdown": np.mean(max_drawdowns) if max_drawdowns else None,
            "Average Strike Crossings": np.mean(strike_crossings) if max_drawdowns else None,
            "Maximum Strike Crossings": np.max(strike_crossings) if max_drawdowns else None,
            "Volatility Clustering": autocorr if autocorr != 0.0 else None
        }
    }

    normalization_factor = max(asian_payoff)
    my_payoff_func = lambda i: asian_payoff[i]/normalization_factor
    

    target_wires = range(n_disc+1)
    estimation_wires = range(n_disc+1, n_disc+n_pe+1)
    
    dev = qml.device("lightning.qubit", wires=(n_disc+n_pe+1))    
    @qml.qnode(dev)
    def circuit():
        qml.templates.QuantumMonteCarlo(
            p,
            my_payoff_func,
            target_wires=target_wires,
            estimation_wires=estimation_wires,
        )

        return qml.probs(estimation_wires)
    
    phase_estimated = np.argmax(circuit()[:int(N_pe / 2)]) / N_pe
    
    estimated_value_qmc = (1 - np.cos(np.pi * phase_estimated)) / 2 * normalization_factor

    # Return both the QMC value and the metrics
    return estimated_value_qmc, metrics

def generate_training_data(num_iterations=1000):
    """Generate training data from multiple QMC simulations"""
    data = []
    all_metrics = []
    
    # Parameter ranges for training
    S0_range = np.linspace(80, 120, 10)
    K_range = np.linspace(90, 110, 10)
    T_range = np.linspace(10/252, 60/252, 10)
    r_range = np.linspace(0.02, 0.08, 10)
    sigma_range = np.linspace(0.1, 0.3, 10)
    
    # Add progress bar with tqdm
    for i in tqdm(range(num_iterations), desc="Generating QMC data"):
        # Randomly sample parameters
        S0 = np.random.choice(S0_range)
        K = np.random.choice(K_range)
        T = np.random.choice(T_range)
        r = np.random.choice(r_range)
        sigma = np.random.choice(sigma_range)
        
        # Run QMC simulation
        result, metrics = Quantum_Monte_Carlo(
            spot=S0, strike=K, time=T, 
            rate=r, volatility=sigma,
            n_disc=6, n_pe=12
        )
        
        data.append({
            'input_S0': S0,
            'input_K': K,
            'input_T': T,
            'input_r': r,
            'input_sigma': sigma,
            'qmc_price': result
        })
        all_metrics.append(metrics)
    
    # After all simulations, print the aggregated metrics
    print("\n=== Simulation Results Summary ===")
    print_metrics_summary(all_metrics)
    
    return pd.DataFrame(data)

def print_metrics_summary(all_metrics):
    """Helper function to print summary of all metrics"""
    print("\n--- Average Option Price Metrics ---")
    avg_price = np.mean([m["Option Price"]["Estimated Price"] for m in all_metrics])
    avg_prob_profit = np.mean([m["Option Price"]["Probability of Profit"] for m in all_metrics])
    print(f"Average Estimated Price: {avg_price:.4f}")
    print(f"Average Probability of Profit: {avg_prob_profit:.2%}")

    print("\n--- Average Greeks ---")
    avg_delta = np.mean([m["Greeks"]["Delta"] for m in all_metrics])
    avg_gamma = np.mean([m["Greeks"]["Gamma"] for m in all_metrics])
    print(f"Average Delta: {avg_delta:.4f}")
    print(f"Average Gamma: {avg_gamma:.4f}")

    print("\n--- Average Path-Dependent Statistics ---")
    avg_drawdown = np.mean([m["Path-Dependent Statistics"]["Average Maximum Drawdown"] 
                           for m in all_metrics if m["Path-Dependent Statistics"]["Average Maximum Drawdown"] is not None])
    print(f"Average Maximum Drawdown: {avg_drawdown:.2%}")

    print("\n--- Stress Test Summary ---")
    for scenario in all_metrics[0]["Stress Test Results"].keys():
        avg_change = np.mean([m["Stress Test Results"][scenario]["Change"] for m in all_metrics])
        print(f"{scenario}: Average Change: {avg_change:+.1f}%")

def train_qmc_model(df):
    """Train ML model on QMC results"""
    # Prepare features and target
    features = ['input_S0', 'input_K', 'input_T', 'input_r', 'input_sigma']
    targets = ['qmc_price', 'std_dev', 'var_95', 'es_95']  # Changed from single target to list of targets
    
    X = df[features]
    y = df[targets]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nModel Performance:")
    print(f"R² Score: {score:.4f}")
    print("\nFeature Importance:")
    print(importance)
    
    return model

def analyze_qmc_patterns(df):
    """Analyze patterns in QMC results"""
    # Correlation analysis
    correlation_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of QMC Results')
    plt.tight_layout()
    plt.show()
    
    # Distribution analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(df['qmc_price'])
    plt.title('Distribution of QMC Prices')
    plt.xlabel('Price')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate training data
    print("Generating training data...")
    df = generate_training_data()  # Reduced for testing
    
    # Train ML model
    print("\nTraining ML model...")
    model = train_qmc_model(df)
    
    # Analyze patterns
    print("\nAnalyzing patterns...")
    analyze_qmc_patterns(df)
    
    # Example prediction
    example_input = pd.DataFrame({
        'input_S0': [100],
        'input_K': [105],
        'input_T': [30/252],
        'input_r': [0.05],
        'input_sigma': [0.2]
    })
    
    prediction = model.predict(example_input)[0]
    print(f"\nExample prediction for standard inputs:")
    print(f"Predicted QMC price: {prediction:.4f}")