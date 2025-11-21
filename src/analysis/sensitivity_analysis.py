"""Sensitivity analysis for portfolio optimization"""

import numpy as np
import pandas as pd
from typing import List, Dict


class SensitivityAnalyzer:
    """Performs sensitivity analysis on portfolio parameters"""
    
    def __init__(self, optimizer):
        """
        Initialize sensitivity analyzer
        
        Args:
            optimizer: PortfolioOptimizer instance
        """
        self.optimizer = optimizer
        self.returns = optimizer.returns
        
    def analyze_risk_free_rate_sensitivity(self, 
                                          rates: List[float]) -> pd.DataFrame:
        """
        Analyze how optimal portfolio changes with risk-free rate
        
        Args:
            rates: List of risk-free rates to test
            
        Returns:
            DataFrame with results for each rate
        """
        results = []
        
        for rate in rates:
            self.optimizer.risk_free_rate = rate
            optimal = self.optimizer.optimize_sharpe_ratio()
            
            results.append({
                'risk_free_rate': rate,
                'expected_return': optimal['expected_return'],
                'volatility': optimal['volatility'],
                'sharpe_ratio': optimal['sharpe_ratio']
            })
        
        return pd.DataFrame(results)
    
    def analyze_weight_sensitivity(self, 
                                  base_weights: Dict[str, float],
                                  asset: str,
                                  weight_range: np.ndarray) -> pd.DataFrame:
        """
        Analyze portfolio metrics as a function of a single asset's weight
        
        Args:
            base_weights: Base portfolio weights
            asset: Asset to vary
            weight_range: Array of weights to test for the asset
            
        Returns:
            DataFrame with portfolio metrics for each weight
        """
        results = []
        assets = list(base_weights.keys())
        
        for weight in weight_range:
            # Adjust other weights proportionally
            remaining_weight = 1.0 - weight
            weights_dict = {a: base_weights[a] for a in assets}
            
            # Calculate sum of other weights
            other_weights_sum = sum(base_weights[a] for a in assets if a != asset)
            
            if other_weights_sum == 0:
                continue
                
            # Distribute remaining weight proportionally
            for a in assets:
                if a == asset:
                    weights_dict[a] = weight
                else:
                    weights_dict[a] = base_weights[a] * remaining_weight / other_weights_sum
            
            # Calculate metrics
            weights_array = np.array([weights_dict[a] for a in assets])
            ret, vol, sharpe = self.optimizer.calculate_portfolio_metrics(weights_array)
            
            results.append({
                f'{asset}_weight': weight,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            })
        
        return pd.DataFrame(results)
    
    def analyze_correlation_impact(self, 
                                  correlation_multipliers: List[float]) -> pd.DataFrame:
        """
        Analyze impact of changing correlation structure on optimal portfolio
        
        Args:
            correlation_multipliers: List of multipliers for correlation matrix
            
        Returns:
            DataFrame with results for each multiplier
        """
        results = []
        original_cov = self.optimizer.cov_matrix.copy()
        
        for multiplier in correlation_multipliers:
            # Extract volatilities
            vols = np.sqrt(np.diag(original_cov))
            
            # Get correlation matrix
            corr = original_cov / np.outer(vols, vols)
            
            # Modify correlations
            np.fill_diagonal(corr, 1.0)  # Keep diagonal as 1
            corr_modified = corr * multiplier
            np.fill_diagonal(corr_modified, 1.0)
            
            # Recreate covariance matrix
            self.optimizer.cov_matrix = corr_modified * np.outer(vols, vols)
            
            try:
                optimal = self.optimizer.optimize_sharpe_ratio()
                results.append({
                    'correlation_multiplier': multiplier,
                    'expected_return': optimal['expected_return'],
                    'volatility': optimal['volatility'],
                    'sharpe_ratio': optimal['sharpe_ratio']
                })
            except:
                continue
        
        # Restore original covariance matrix
        self.optimizer.cov_matrix = original_cov
        
        return pd.DataFrame(results)
    
    def monte_carlo_sensitivity(self, 
                               base_weights: np.ndarray,
                               num_simulations: int = 1000) -> pd.DataFrame:
        """
        Perform Monte Carlo simulation on portfolio weights
        
        Args:
            base_weights: Base portfolio weights
            num_simulations: Number of random portfolios to generate
            
        Returns:
            DataFrame with simulated portfolio metrics
        """
        results = []
        
        for _ in range(num_simulations):
            # Generate random weights
            random_weights = np.random.random(len(base_weights))
            random_weights /= random_weights.sum()
            
            # Calculate metrics
            ret, vol, sharpe = self.optimizer.calculate_portfolio_metrics(random_weights)
            
            results.append({
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            })
        
        return pd.DataFrame(results)
