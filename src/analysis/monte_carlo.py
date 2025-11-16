"""Monte Carlo simulation for portfolio analysis"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio forecasting"""
    
    def __init__(self, returns: pd.DataFrame, initial_investment: float = 10000):
        """
        Initialize Monte Carlo simulator
        
        Args:
            returns: DataFrame of historical daily returns
            initial_investment: Initial portfolio value
        """
        self.returns = returns
        self.initial_investment = initial_investment
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        
    def simulate_portfolio(self, weights: np.ndarray, 
                          num_simulations: int = 10000,
                          num_days: int = 252) -> np.ndarray:
        """
        Run Monte Carlo simulation for a portfolio
        
        Args:
            weights: Portfolio weights (must sum to 1)
            num_simulations: Number of simulation paths
            num_days: Number of trading days to simulate
            
        Returns:
            Array of shape (num_simulations, num_days) with portfolio values
        """
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("Weights must sum to 1")
            
        # Portfolio statistics
        portfolio_mean = np.dot(weights, self.mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Generate random returns
        simulations = np.zeros((num_simulations, num_days))
        
        for i in range(num_simulations):
            # Generate daily returns using normal distribution
            daily_returns = np.random.normal(portfolio_mean, portfolio_std, num_days)
            
            # Calculate cumulative portfolio value
            price_series = self.initial_investment * (1 + daily_returns).cumprod()
            simulations[i] = price_series
            
        return simulations
    
    def get_simulation_statistics(self, simulations: np.ndarray) -> dict:
        """
        Calculate statistics from simulation results
        
        Args:
            simulations: Array of simulation results
            
        Returns:
            Dictionary with statistical measures
        """
        final_values = simulations[:, -1]
        
        stats = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'probability_of_loss': np.sum(final_values < self.initial_investment) / len(final_values),
            'expected_return': (np.mean(final_values) - self.initial_investment) / self.initial_investment
        }
        
        return stats
    
    def calculate_var_cvar(self, simulations: np.ndarray, 
                           confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        
        Args:
            simulations: Array of simulation results
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        final_values = simulations[:, -1]
        returns = (final_values - self.initial_investment) / self.initial_investment
        
        # VaR: maximum loss at given confidence level
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # CVaR: expected loss beyond VaR
        cvar = returns[returns <= var].mean()
        
        return var, cvar
