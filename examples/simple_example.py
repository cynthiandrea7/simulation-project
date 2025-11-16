#!/usr/bin/env python3
"""Simple example demonstrating the core functionality"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.stock_data import StockDataCollector
from analysis.portfolio_optimization import PortfolioOptimizer
from analysis.monte_carlo import MonteCarloSimulator
import numpy as np


def main():
    print("=" * 60)
    print("  Simple Quantitative Financial Analysis Example")
    print("=" * 60)
    
    # 1. Collect data
    print("\n1. Collecting stock data...")
    collector = StockDataCollector(['AAPL', 'MSFT', 'GOOGL'])
    data = collector.fetch_data(period='1y')
    returns = collector.calculate_returns()
    print(f"   Fetched {len(data)} days of data for 3 stocks")
    
    # 2. Optimize portfolio
    print("\n2. Optimizing portfolio...")
    optimizer = PortfolioOptimizer(returns)
    optimal = optimizer.optimize_sharpe_ratio()
    print(f"   Optimal Sharpe Ratio: {optimal['sharpe_ratio']:.4f}")
    print("   Optimal Weights:")
    for ticker, weight in optimal['weights'].items():
        print(f"     {ticker}: {weight:.2%}")
    
    # 3. Run Monte Carlo simulation
    print("\n3. Running Monte Carlo simulation...")
    mc = MonteCarloSimulator(returns, initial_investment=10000)
    weights = np.array([optimal['weights'][ticker] for ticker in returns.columns])
    simulations = mc.simulate_portfolio(weights, num_simulations=1000, num_days=252)
    stats = mc.get_simulation_statistics(simulations)
    
    print(f"   Expected final value: ${stats['mean_final_value']:,.2f}")
    print(f"   Expected return: {stats['expected_return']:.2%}")
    print(f"   Probability of loss: {stats['probability_of_loss']:.2%}")
    
    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
