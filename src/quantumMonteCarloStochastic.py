import pennylane as qml
import numpy as np
import scipy
import scipy.stats
from time import time
import matplotlib.pyplot as plt


def payoff_func(spot, strike, rate, volatility, t, x, dt):
    all_paths = []
    stock_prices = []

    for W in x:
        price = [spot]
        price.append(price[-1] * np.exp(volatility * W + (rate - 0.5 * volatility ** 2) * dt))
        price_array = np.array(price)
        all_paths.append(price_array)
        stock_prices.append(price_array[-1])  # Just take the final stock price

    return np.array(stock_prices), np.array(all_paths)


def normal_dist(n, variance, cutoff_factor):
    dim = 2 ** n
    points = np.random.normal(0, np.sqrt(variance), dim)
    points = np.clip(points, -cutoff_factor * np.sqrt(variance), cutoff_factor * np.sqrt(variance))
    prob = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(- 0.5 * points ** 2 / variance)
    prob_renorm = prob / np.sum(prob)
    return [points, prob_renorm]


def Quantum_Monte_Carlo(T, dt, spot=100, strike=100, rate=0.05, volatility=0.2, t=1, variance=1.0, cutoff_factor=4, n_disc=4, n_pe=10):
    N_disc, N_pe = 2**n_disc, 2**n_pe
    
    x, p = normal_dist(n_disc, variance, cutoff_factor)
    stock_prices, paths = payoff_func(spot, strike, rate, volatility, t, x, dt)
    
    # Calculate stock price statistics
    mean_price = np.mean(stock_prices)
    std_dev = np.std(stock_prices)
    
    # Calculate confidence interval
    z_score = 1.96  # 95% confidence level
    margin_error = z_score * (std_dev / np.sqrt(len(stock_prices)))
    confidence_interval = (mean_price - margin_error, mean_price + margin_error)
    
    # Calculate higher moments
    skewness = np.mean(((stock_prices - mean_price) / std_dev) ** 3)
    kurtosis = np.mean(((stock_prices - mean_price) / std_dev) ** 4) - 3
    
    # Modify probability calculations for stock price
    prob_increase = np.mean(stock_prices > spot)
    max_loss = spot - np.min(stock_prices)
    max_gain = np.max(stock_prices) - spot
    
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
        stress_price = payoff_func(test_spot, strike, test_rate, test_vol, t, x, dt)[0].mean()
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
    price_up = payoff_func(spot + delta_shift, strike, rate, volatility, t, x, dt)[0].mean()
    price_down = payoff_func(spot - delta_shift, strike, rate, volatility, t, x, dt)[0].mean()
    delta = (price_up - price_down) / (2 * delta_shift)
    

    # Gamma calculation (d²V/dS²)
    gamma = (price_up - 2*mean_price + price_down) / (delta_shift**2)
    
    # Theta calculation (dV/dt)
    if t > theta_shift:
        price_next = payoff_func(spot, strike, rate, volatility, t - theta_shift, x, dt)[0].mean()
        theta = (price_next - mean_price) / theta_shift
    else:

        theta = 0
    
    print("\n--- Additional Inference Statistics ---")
    print(f"Estimated Stock Price: {mean_price:.4f}")
    if not np.isnan(std_dev):
        print(f"95% Confidence Interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
        print(f"Standard Deviation: {std_dev:.4f}")
        print(f"Skewness: {skewness:.4f}")
        print(f"Kurtosis: {kurtosis:.4f}")
    print(f"Probability of Increase: {prob_increase:.2%}")
    print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")

    # Print Greeks
    print("\n--- Greeks ---")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    if t > theta_shift:
        print(f"Theta (daily): {theta:.4f}")

    
    print("\n--- Stress Test Results ---")
    for scenario, price in stress_results.items():
        pct_change = ((price/mean_price - 1) * 100)
        print(f"{scenario}: {price:.4f} (Change: {pct_change:+.1f}%)")
    
    # Add checks for path-dependent statistics
    if max_drawdowns:
        print("\n--- Path-Dependent Statistics ---")
        print(f"Average Maximum Drawdown: {np.mean(max_drawdowns):.2%}")
        print(f"Average Strike Crossings: {np.mean(strike_crossings):.2f}")
        print(f"Maximum Strike Crossings: {np.max(strike_crossings):.0f}")
    
    if autocorr != 0.0:
        print(f"\nVolatility Clustering (Return Autocorrelation): {autocorr:.4f}")
    
    normalization_factor = max(stock_prices)
    my_payoff_func = lambda i: stock_prices[i]/normalization_factor
    

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
    
    
    # Plot the paths
    plt.figure(figsize=(10, 6))
    t_points = np.linspace(0, t, 2)  # 2 points for start and end
    for path in paths:
        plt.scatter(t_points, path, alpha=0.5)
    plt.title('Monte Carlo Simulation Paths')
    plt.xlabel('t (years)')
    plt.ylabel('Stock Price')
    #plt.grid(True)
    # Add mean path line
    mean_path = np.mean([path for path in paths], axis=0)
    plt.plot(t_points, mean_path, 'r-', linewidth=2, label='Mean Path')
    
    # Add strike price reference line
    plt.axhline(y=strike, color='g', linestyle='--', alpha=0.5, label='Strike Price')
    
    # Add strike price annotation
    plt.annotate(f'Strike: ${strike:.2f}', 
                xy=(t_points[-1], strike),
                xytext=(t_points[-1]+0.05, strike),
                fontsize=8,
                color='g')
    
    # Add confidence bands
    std_path = np.std([path for path in paths], axis=0)
    plt.fill_between(t_points, 
                     mean_path - 2*std_path,
                     mean_path + 2*std_path,
                     alpha=0.2,
                     color='gray',
                     label='95% Confidence Band')
    
    # Enhance aesthetics
    plt.title('Quantum Monte Carlo Simulation Paths', fontsize=12, pad=15)
    plt.xlabel('Time (years)', fontsize=10)
    plt.ylabel('Stock Price ($)', fontsize=10)
    plt.legend(loc='upper left')
    
    # Add price annotations
    plt.annotate(f'Start: ${spot:.2f}', 
                xy=(t_points[0], spot),
                xytext=(t_points[0]-0.1, spot+2),
                fontsize=8)

    plt.annotate(f'Mean End: ${mean_path[-1]:.2f}',
                xy=(t_points[-1], mean_path[-1]),
                xytext=(t_points[-1]-0.1, mean_path[-1]+2),
                fontsize=8)
                
    plt.tight_layout()

    
    # Save plot to SVG file
    current_t = round(time())
    filename = f"QMC_{current_t}.svg"
    plt.savefig(f"src/static/images/graphs/{filename}", format='svg', bbox_inches='tight')
    plt.close()

    # Create dictionary to store all statistics
    stats_dict = {
        'inference_statistics': {
            'estimated_price': round(float(mean_price), 2),
            'confidence_interval': {
                'lower': confidence_interval[0],
                'upper': confidence_interval[1]
            },
            'standard_deviation': float(std_dev),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'probability_of_profit': float(prob_increase),
            'risk_reward_ratio': float(risk_reward_ratio)
        },
        'greeks': {
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta) if t > theta_shift else None
        },
        'stress_test_results': {
            scenario: {
                'price': float(price),
                'percent_change': float(((price/mean_price - 1) * 100))
            } for scenario, price in stress_results.items()
        },
        'path_dependent_stats': {
            'average_max_drawdown': float(np.mean(max_drawdowns)) if max_drawdowns else None,
            'average_strike_crossings': float(np.mean(strike_crossings)) if max_drawdowns else None,
            'maximum_strike_crossings': float(np.max(strike_crossings)) if max_drawdowns else None,
            'volatility_clustering': float(autocorr) if autocorr != 0.0 else None
        }
    }

    return np.array(estimated_value_qmc), filename, stats_dict

def Q_sim_start(S0, K, T, r, sigma ):

    # --- Step 1: Define parameters ---
    T = 5*T/252 # (typical US market)
    trading_hours = 6.5  # Trading hours per day (typical US market)
    num_steps = int(30 * trading_hours)  # Number of hourly steps (6.5 hours * 30 days)
    dt = T / num_steps # t step

    
    results, filename, stats = Quantum_Monte_Carlo(n_disc=6, n_pe=12, T=T, dt=dt, spot=S0, strike=K)
    return filename, stats



# if __name__ == "__main__":
#     print(Q_sim_start(100, 105, 4, 0.05, 0.2))
