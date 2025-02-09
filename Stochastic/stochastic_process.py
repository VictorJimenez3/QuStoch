import numpy as np
import matplotlib.pyplot as plt


T = 1  # Total time
N = 1000000 # Number of time steps
dt = T / N  # Time step size
num_paths = 5  # Number of sample paths


t = np.linspace(0, T, N+1)

# Initialize Wiener Process
W = np.zeros((num_paths, N+1))

# For reproducibility
np.random.seed(42) 

for i in range(num_paths):
    dW = np.sqrt(dt) * np.random.randn(N)  # Gaussian increment
    print(dW)
    W[i, 1: ] = np.cumsum(dW)  # Cumulative sum to get W_t



plt.figure(figsize=(10, 6))
for i in range(num_paths):
    plt.plot(t, W[i], label=f'Path {i+1}')

plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Reference line at W=0
plt.xlabel('Time (t)')
plt.ylabel('Wiener Process W_t')
plt.title('Multiple Sample Paths of the Wiener Process')
plt.legend()
plt.show()
