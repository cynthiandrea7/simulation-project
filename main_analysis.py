#!/usr/bin/env python3
"""Main script for Quantitative Financial Analysis"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.stock_data import StockDataCollector
from analysis.monte_carlo import MonteCarloSimulator
from analysis.portfolio_optimization import PortfolioOptimizer
from analysis.sensitivity_analysis import SensitivityAnalyzer
from visualization.plots import (
    plot_price_history, plot_returns_distribution, plot_correlation_heatmap,
    plot_monte_carlo_simulation, plot_final_value_distribution,
    plot_efficient_frontier, plot_portfolio_weights, plot_sensitivity_analysis
)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Run complete quantitative financial analysis"""
    
    print_section("QUANTITATIVE FINANCIAL ANALYSIS")
    print("Analyzing the Magnificent Seven stocks")
    print("Stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META\n")
    
    # Configuration
    INITIAL_INVESTMENT = 10000
    NUM_SIMULATIONS = 10000
    NUM_DAYS = 252  # One trading year
    OUTPUT_DIR = "output"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==================== DATA COLLECTION ====================
    print_section("1. DATA COLLECTION")
    
    collector = StockDataCollector()
    print("Fetching 2 years of historical data...")
    data = collector.fetch_data(period='2y')
    returns = collector.calculate_returns()
    
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    stats = collector.get_summary_statistics()
    print(stats.to_string())
    
    # Correlation matrix
    print("\n" + "-" * 80)
    print("CORRELATION MATRIX")
    print("-" * 80)
    corr_matrix = collector.get_correlation_matrix()
    print(corr_matrix.to_string())
    
    # Visualizations
    plot_price_history(data, save_path=f"{OUTPUT_DIR}/price_history.png")
    plot_returns_distribution(returns, save_path=f"{OUTPUT_DIR}/returns_distribution.png")
    plot_correlation_heatmap(corr_matrix, save_path=f"{OUTPUT_DIR}/correlation_heatmap.png")
    print("\n✓ Saved: price_history.png, returns_distribution.png, correlation_heatmap.png")
    
    # ==================== PORTFOLIO OPTIMIZATION ====================
    print_section("2. PORTFOLIO OPTIMIZATION")
    
    optimizer = PortfolioOptimizer(returns, risk_free_rate=0.02)
    
    # Equal weight portfolio
    print("Calculating Equal Weight Portfolio...")
    equal_weight = optimizer.equal_weight_portfolio()
    print("\nEqual Weight Portfolio:")
    print(f"  Expected Return: {equal_weight['expected_return']:.2%}")
    print(f"  Volatility: {equal_weight['volatility']:.2%}")
    print(f"  Sharpe Ratio: {equal_weight['sharpe_ratio']:.4f}")
    
    # Maximum Sharpe ratio portfolio
    print("\nOptimizing for Maximum Sharpe Ratio...")
    max_sharpe = optimizer.optimize_sharpe_ratio()
    print("\nMaximum Sharpe Ratio Portfolio:")
    print(f"  Expected Return: {max_sharpe['expected_return']:.2%}")
    print(f"  Volatility: {max_sharpe['volatility']:.2%}")
    print(f"  Sharpe Ratio: {max_sharpe['sharpe_ratio']:.4f}")
    print("\n  Optimal Weights:")
    for asset, weight in sorted(max_sharpe['weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {asset}: {weight:.2%}")
    
    # Minimum variance portfolio
    print("\nOptimizing for Minimum Variance...")
    min_var = optimizer.optimize_minimum_variance()
    print("\nMinimum Variance Portfolio:")
    print(f"  Expected Return: {min_var['expected_return']:.2%}")
    print(f"  Volatility: {min_var['volatility']:.2%}")
    print(f"  Sharpe Ratio: {min_var['sharpe_ratio']:.4f}")
    print("\n  Optimal Weights:")
    for asset, weight in sorted(min_var['weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {asset}: {weight:.2%}")
    
    # Efficient frontier
    print("\nGenerating Efficient Frontier...")
    efficient_frontier = optimizer.efficient_frontier(num_portfolios=100)
    
    # Visualizations
    plot_efficient_frontier(efficient_frontier, max_sharpe, min_var, equal_weight,
                           save_path=f"{OUTPUT_DIR}/efficient_frontier.png")
    plot_portfolio_weights(max_sharpe['weights'], 
                          title="Maximum Sharpe Ratio Portfolio Weights",
                          save_path=f"{OUTPUT_DIR}/optimal_weights.png")
    plot_portfolio_weights(equal_weight['weights'],
                          title="Equal Weight Portfolio Weights",
                          save_path=f"{OUTPUT_DIR}/equal_weights.png")
    print("\n✓ Saved: efficient_frontier.png, optimal_weights.png, equal_weights.png")
    
    # ==================== MONTE CARLO SIMULATION ====================
    print_section("3. MONTE CARLO SIMULATION")
    
    mc_simulator = MonteCarloSimulator(returns, initial_investment=INITIAL_INVESTMENT)
    
    # Run simulation for optimal portfolio
    print(f"Running {NUM_SIMULATIONS:,} Monte Carlo simulations...")
    print(f"Simulation period: {NUM_DAYS} trading days (1 year)")
    print(f"Initial investment: ${INITIAL_INVESTMENT:,}")
    
    optimal_weights = np.array([max_sharpe['weights'][ticker] for ticker in returns.columns])
    simulations = mc_simulator.simulate_portfolio(optimal_weights, NUM_SIMULATIONS, NUM_DAYS)
    
    # Get statistics
    sim_stats = mc_simulator.get_simulation_statistics(simulations)
    var, cvar = mc_simulator.calculate_var_cvar(simulations, confidence_level=0.95)
    
    print("\n" + "-" * 80)
    print("SIMULATION RESULTS")
    print("-" * 80)
    print(f"Mean Final Value: ${sim_stats['mean_final_value']:,.2f}")
    print(f"Median Final Value: ${sim_stats['median_final_value']:,.2f}")
    print(f"Std Dev: ${sim_stats['std_final_value']:,.2f}")
    print(f"Min: ${sim_stats['min_final_value']:,.2f}")
    print(f"Max: ${sim_stats['max_final_value']:,.2f}")
    print(f"\nPercentiles:")
    print(f"  5th: ${sim_stats['percentile_5']:,.2f}")
    print(f"  25th: ${sim_stats['percentile_25']:,.2f}")
    print(f"  75th: ${sim_stats['percentile_75']:,.2f}")
    print(f"  95th: ${sim_stats['percentile_95']:,.2f}")
    print(f"\nRisk Metrics:")
    print(f"  Probability of Loss: {sim_stats['probability_of_loss']:.2%}")
    print(f"  Expected Return: {sim_stats['expected_return']:.2%}")
    print(f"  VaR (95%): {var:.2%}")
    print(f"  CVaR (95%): {cvar:.2%}")
    
    # Visualizations
    plot_monte_carlo_simulation(simulations, INITIAL_INVESTMENT,
                               save_path=f"{OUTPUT_DIR}/monte_carlo_simulation.png")
    plot_final_value_distribution(simulations, INITIAL_INVESTMENT,
                                 save_path=f"{OUTPUT_DIR}/final_value_distribution.png")
    print("\n✓ Saved: monte_carlo_simulation.png, final_value_distribution.png")
    
    # ==================== SENSITIVITY ANALYSIS ====================
    print_section("4. SENSITIVITY ANALYSIS")
    
    sensitivity = SensitivityAnalyzer(optimizer)
    
    # Risk-free rate sensitivity
    print("Analyzing Risk-Free Rate Sensitivity...")
    rates = np.linspace(0.0, 0.05, 11)
    rate_sensitivity = sensitivity.analyze_risk_free_rate_sensitivity(rates.tolist())
    print("\nRisk-Free Rate Impact on Optimal Portfolio:")
    print(rate_sensitivity.to_string(index=False))
    
    plot_sensitivity_analysis(rate_sensitivity, 'risk_free_rate', 'sharpe_ratio',
                            'Sharpe Ratio vs Risk-Free Rate',
                            save_path=f"{OUTPUT_DIR}/rate_sensitivity.png")
    
    # Weight sensitivity for top holding
    print("\nAnalyzing Weight Sensitivity for Top Holdings...")
    top_asset = max(max_sharpe['weights'].items(), key=lambda x: x[1])[0]
    weight_range = np.linspace(0.0, 0.5, 21)
    weight_sensitivity = sensitivity.analyze_weight_sensitivity(
        max_sharpe['weights'], top_asset, weight_range
    )
    
    plot_sensitivity_analysis(weight_sensitivity, f'{top_asset}_weight', 'sharpe_ratio',
                            f'Sharpe Ratio vs {top_asset} Weight',
                            save_path=f"{OUTPUT_DIR}/weight_sensitivity.png")
    print(f"\n✓ Saved: rate_sensitivity.png, weight_sensitivity.png")
    
    # ==================== SUMMARY ====================
    print_section("ANALYSIS COMPLETE")
    
    print("Key Findings:")
    print(f"\n1. Optimal Portfolio (Max Sharpe Ratio):")
    print(f"   - Expected Annual Return: {max_sharpe['expected_return']:.2%}")
    print(f"   - Annual Volatility: {max_sharpe['volatility']:.2%}")
    print(f"   - Sharpe Ratio: {max_sharpe['sharpe_ratio']:.4f}")
    
    print(f"\n2. Monte Carlo Simulation (1-year forecast):")
    print(f"   - Expected Portfolio Value: ${sim_stats['mean_final_value']:,.2f}")
    print(f"   - Expected Return: {sim_stats['expected_return']:.2%}")
    print(f"   - Probability of Loss: {sim_stats['probability_of_loss']:.2%}")
    print(f"   - Value at Risk (95%): {var:.2%}")
    
    print(f"\n3. Risk Management:")
    print(f"   - Diversification across {len(collector.tickers)} stocks")
    print(f"   - Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    print(f"   - CVaR (95%): {cvar:.2%} (expected loss in worst 5% scenarios)")
    
    print(f"\nAll results and visualizations saved to '{OUTPUT_DIR}/' directory")
    print("\nGenerated files:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {file}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
