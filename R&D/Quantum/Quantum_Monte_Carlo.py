import pennylane as qml
import numpy as np
import scipy
import scipy.stats
from time import time
import matplotlib.pyplot as plt


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
    normalization_factor = max(asian_payoff)
    my_payoff_func = lambda i: asian_payoff[i]/normalization_factor
    
    # Plot the paths and save to PNG
    plt.figure(figsize=(10, 6))
    time_points = np.linspace(0, time, 2)  # 2 points for start and end
    for path in paths:
        plt.scatter(time_points, path, alpha=0.5)
    plt.title('Monte Carlo Simulation Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.savefig('monte_carlo_paths.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


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
        
    return np.array(estimated_value_qmc)

t1 = time()
results = Quantum_Monte_Carlo(n_disc=6, n_pe=12)
t2 = time()
print('Results:', results, "time:", t2 - t1)