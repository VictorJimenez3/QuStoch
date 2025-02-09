import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define parameters ---
S0 = 100.0     # Current stock price
K = 105.0      # Strike price
T = 1.0        # Time to maturity (in years)
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility of the underlying asset
num_sims = 10_000  # Number of simulated paths
num_steps = 252    # Number of time steps (e.g., ~252 trading days in a year)

# --- Step 2: Prepare to store price paths ---
# We'll create a 2D array where:
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

# --- Step 5: Discount the payoff to get the present value ---
option_price = np.exp(-r * T) * np.mean(payoffs)
print("Estimated European Call Option Price:", option_price)

# --- Step 6: Plot a subset of the simulated paths ---
plt.figure(figsize=(10, 6))
num_paths_to_plot = 10
for i in range(num_paths_to_plot):
    plt.plot(S[:, i], lw=1)
plt.title("Sample of Monte Carlo Stock Price Paths")
plt.xlabel("Time Steps (Days)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()
