import numpy as np
import matplotlib.pyplot as plt



# Parameters for Geometric Brownian Motion
S0 = 100       # Initial stock price
mu = -0.05      # Drift (expected return)
sigma = 0.2    # Volatility
T = 1          # Total time (in years)
N = 10       # Number of time steps
dt = T / N     # Time step size

# Time vector
t = np.linspace(0, T, N+1)

# Generate Wiener process increments and cumulative sum
#np.random.seed(42)  # For reproducibility
dW = np.sqrt(dt) * np.random.randn(N)  # Random increments
W = np.cumsum(dW)  # Wiener process


# Model GBM using the given formula:
#  S_t = S0 * exp((mu - sigma^2 / 2) * t + sigma * W_t)

drift = (mu - 0.5 * sigma**2) * t[1:]  # Drift term
diffusion = sigma * W  # Random term from Wiener process

S = S0 * np.exp(np.insert(drift + diffusion, 0, 0))  # Stock price with initial S0

print(S)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Simulated GBM')
plt.axhline(S0, color='black', linestyle='--', linewidth=1, label='Initial Price $100')
plt.xlabel('Time (t)')
plt.ylabel('Stock Price (S_t)')
plt.title('Geometric Brownian Motion Simulation')
plt.legend()
plt.show()
