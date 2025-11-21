"""Portfolio optimization and risk analysis"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory"""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer
        
        Args:
            returns: DataFrame of historical daily returns
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252  # Annualized
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns.columns)
        
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_sharpe_ratio(self) -> dict:
        """
        Find portfolio with maximum Sharpe ratio
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        # Objective: minimize negative Sharpe ratio
        def objective(weights):
            _, _, sharpe = self.calculate_portfolio_metrics(weights)
            return -sharpe
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: each weight between 0 and 1
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1.0 / self.num_assets] * self.num_assets)
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise ValueError("Optimization failed")
            
        optimal_weights = result.x
        ret, vol, sharpe = self.calculate_portfolio_metrics(optimal_weights)
        
        return {
            'weights': dict(zip(self.returns.columns, optimal_weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    
    def optimize_minimum_variance(self) -> dict:
        """
        Find minimum variance portfolio
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: each weight between 0 and 1
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1.0 / self.num_assets] * self.num_assets)
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise ValueError("Optimization failed")
            
        optimal_weights = result.x
        ret, vol, sharpe = self.calculate_portfolio_metrics(optimal_weights)
        
        return {
            'weights': dict(zip(self.returns.columns, optimal_weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    
    def efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            num_portfolios: Number of portfolios to generate
            
        Returns:
            DataFrame with portfolio metrics
        """
        results = []
        
        # Generate target returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        for target_return in target_returns:
            try:
                # Objective: minimize volatility
                def objective(weights):
                    return np.dot(weights.T, np.dot(self.cov_matrix, weights))
                
                # Constraints
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - target_return}
                ]
                
                # Bounds
                bounds = tuple((0, 1) for _ in range(self.num_assets))
                
                # Initial guess
                initial_guess = np.array([1.0 / self.num_assets] * self.num_assets)
                
                # Optimize
                result = minimize(objective, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                
                if result.success:
                    ret, vol, sharpe = self.calculate_portfolio_metrics(result.x)
                    results.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe_ratio': sharpe
                    })
            except:
                continue
        
        return pd.DataFrame(results)
    
    def equal_weight_portfolio(self) -> dict:
        """
        Calculate metrics for equal-weighted portfolio
        
        Returns:
            Dictionary with equal weights and metrics
        """
        weights = np.array([1.0 / self.num_assets] * self.num_assets)
        ret, vol, sharpe = self.calculate_portfolio_metrics(weights)
        
        return {
            'weights': dict(zip(self.returns.columns, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
