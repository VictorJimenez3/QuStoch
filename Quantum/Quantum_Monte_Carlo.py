import pennylane as qml
import numpy as np
import scipy
import scipy.stats
from time import time
import matplotlib.pyplot as plt


# --- Step 1: Define parameters ---
S0 = 100.0     # Current stock price
K = 105.0      # Strike price
T = 1  # time (e.g. in years)
r = 0.05  # risk free rate
sigma = 0.2  # volatility
num_steps = 12 # Number of steps
dt = T / num_steps # Time step

variance = 1 # Variance of the Normal Distribution being used.
cutoff_factor = 4


def payoff_func(spot, strike, rate, volatility, time, x):
    payoffs = []

    for W in x:
        price = [spot]
        price.append(price[-1] * np.exp(volatility * W + (rate - 0.5 * volatility ** 2) * dt))
        price = np.expand_dims(np.array(price[1:]), axis=0)
        payoff = call_payoffs(price, strike, spot)[0]
        payoffs.append(payoff)


    if np.max(payoffs) == 0:
        payoffs[0] = 1e-10
    return np.array(payoffs)*np.exp(- rate * time)



def call_payoffs(paths, strike, spot):
    spots = np.full((paths.shape[0], 1), spot)
    paths = np.append(spots, paths, axis=1)
    
    means = scipy.stats.mstats.gmean(paths, axis=1)
    
    asian_payoffs = means - strike
    asian_payoffs[asian_payoffs < 0] = 0

    return asian_payoffs



def normal_dist(n, variance, cutoff_factor):
    dim = 2 ** n
    cutoff_tmp = cutoff_factor * np.sqrt(variance)
    points = np.linspace(-cutoff_tmp, cutoff_tmp, num=dim)
    prob = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(- 0.5 * points ** 2 / variance)
    prob_renorm = prob / np.sum(prob)
    return [points, prob_renorm]






