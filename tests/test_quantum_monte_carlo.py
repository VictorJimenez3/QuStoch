import pytest
import numpy as np
from Quantum_Monte_Carlo import (
    payoff_func,
    call_payoffs,
    normal_dist,
    Quantum_Monte_Carlo
)



def test_payoff_func():
    # Test basic functionality
    spot = 100.0
    strike = 105.0
    rate = 0.05
    volatility = 0.2
    time = 1

    x = np.array([0.1, 0.2])  # Sample Wiener process increments
        
    result = payoff_func(spot, strike, rate, volatility, time, x)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == len(x)
    assert all(r >= 0 for r in result)  # Payoffs should be non-negative
    
    # Test with extreme values

    x_extreme = np.array([2.0, -2.0])  # Large positive and negative increments
    result_extreme = payoff_func(spot, strike, rate, volatility, time, x_extreme)
    assert all(r >= 0 for r in result_extreme)
    

    # Test with zero volatility
    result_zero_vol = payoff_func(spot, strike, rate, 0.0, time, x)
    assert all(np.isfinite(result_zero_vol))
    

    # Test with zero rate
    result_zero_rate = payoff_func(spot, strike, 0.0, volatility, time, x)
    assert all(np.isfinite(result_zero_rate))



def test_call_payoffs():
    # Test with known values
    paths = np.array([[110.0, 120.0]])  # Single path with two steps
    strike = 105.0
    spot = 100.0


    result = call_payoffs(paths, strike, spot)
    

    assert isinstance(result, np.ndarray)
    assert len(result) == paths.shape[0]
    assert all(r >= 0 for r in result)  # Payoffs should be non-negative
    
    # Test when payoff should be zero
    paths_low = np.array([[90.0, 95.0]])
    strike_high = 150.0
    result_zero = call_payoffs(paths_low, strike_high, spot)
    assert all(r == 0 for r in result_zero)
    

    # Test with multiple paths
    multi_paths = np.array([
        [110.0, 120.0],  # Should give positive payoff
        [90.0, 95.0],    # Should give zero payoff
        [150.0, 160.0]   # Should give large positive payoff
    ])
    result_multi = call_payoffs(multi_paths, strike, spot)
    assert len(result_multi) == 3
    assert result_multi[0] > 0  # First path should have positive payoff
    assert result_multi[1] == 0  # Second path should have zero payoff

    assert result_multi[2] > result_multi[0]  # Third path should have larger payoff
    
    # Test with edge case where average equals strike
    paths_edge = np.array([[strike, strike]])
    result_edge = call_payoffs(paths_edge, strike, spot)
    assert all(r == 0 for r in result_edge)



def test_normal_dist():
    # Test with standard parameters
    n = 3  # Will generate 2^3 = 8 points
    variance = 1.0
    cutoff_factor = 4.0
    
    points, probabilities = normal_dist(n, variance, cutoff_factor)
    
    assert isinstance(points, np.ndarray)
    assert isinstance(probabilities, np.ndarray)
    assert len(points) == 2**n
    assert len(probabilities) == 2**n
    assert np.isclose(np.sum(probabilities), 1.0)  # Probabilities should sum to 1
    assert all(p >= 0 for p in probabilities)  # All probabilities should be non-negative
    
    # Test symmetry of the distribution
    assert np.allclose(points[0:len(points)//2], -points[len(points)//2:][::-1])
    
    # Test with different parameters
    n_large = 4  # 16 points
    variance_large = 2.0
    cutoff_large = 5.0
    
    points_large, probs_large = normal_dist(n_large, variance_large, cutoff_large)
    assert len(points_large) == 2**n_large
    assert len(probs_large) == 2**n_large
    assert np.isclose(np.sum(probs_large), 1.0)
    
    # Test with small variance
    points_small, probs_small = normal_dist(n, 0.1, cutoff_factor)
    assert np.max(np.abs(points_small)) < np.max(np.abs(points))  # Points should be closer to zero
    
    # Test mean of the distribution
    assert np.isclose(np.sum(points * probabilities), 0.0, atol=1e-10)  # Mean should be zero
    assert np.isclose(np.sum(points_large * probs_large), 0.0, atol=1e-10) 



def test_quantum_monte_carlo():
    # Test with default parameters
    result = Quantum_Monte_Carlo()
    assert isinstance(result, np.ndarray)
    assert result >= 0  # Option price should be non-negative
    assert np.isfinite(result)  # Result should be finite

    # Test with in-the-money option
    result_itm = Quantum_Monte_Carlo(spot=120, strike=100)
    assert result_itm > 0  # Should have positive value for ITM option

    # Test with out-of-the-money option
    result_otm = Quantum_Monte_Carlo(spot=80, strike=100)
    assert result_otm >= 0  # Should be non-negative for OTM option

    # Test with different discretization and estimation parameters
    result_high_precision = Quantum_Monte_Carlo(n_disc=5, n_pe=11)
    assert isinstance(result_high_precision, np.ndarray)
    assert result_high_precision >= 0

    # Test with edge cases
    result_zero_vol = Quantum_Monte_Carlo(volatility=0.0)
    assert np.isfinite(result_zero_vol)

    result_zero_rate = Quantum_Monte_Carlo(rate=0.0)
    assert np.isfinite(result_zero_rate) 


