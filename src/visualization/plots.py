"""Visualization functions for financial analysis"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional


sns.set_style('whitegrid')


def plot_price_history(data: pd.DataFrame, title: str = "Stock Price History",
                      figsize: tuple = (12, 6), save_path: Optional[str] = None):
    """
    Plot historical stock prices
    
    Args:
        data: DataFrame with stock prices
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    for column in data.columns:
        plt.plot(data.index, data[column], label=column, linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_returns_distribution(returns: pd.DataFrame, 
                             figsize: tuple = (14, 8),
                             save_path: Optional[str] = None):
    """
    Plot distribution of returns for each stock
    
    Args:
        returns: DataFrame with returns
        figsize: Figure size
        save_path: Path to save figure
    """
    num_stocks = len(returns.columns)
    rows = (num_stocks + 2) // 3
    
    fig, axes = plt.subplots(rows, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, column in enumerate(returns.columns):
        axes[idx].hist(returns[column], bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{column} Returns Distribution')
        axes[idx].set_xlabel('Daily Return')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_stocks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame,
                            figsize: tuple = (10, 8),
                            save_path: Optional[str] = None):
    """
    Plot correlation heatmap
    
    Args:
        correlation_matrix: Correlation matrix
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Stock Returns Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_monte_carlo_simulation(simulations: np.ndarray,
                               initial_investment: float,
                               num_paths_to_plot: int = 100,
                               figsize: tuple = (12, 6),
                               save_path: Optional[str] = None):
    """
    Plot Monte Carlo simulation results
    
    Args:
        simulations: Array of simulation paths
        initial_investment: Initial portfolio value
        num_paths_to_plot: Number of paths to display
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    # Plot sample paths
    for i in range(min(num_paths_to_plot, simulations.shape[0])):
        plt.plot(simulations[i], color='blue', alpha=0.1, linewidth=0.5)
    
    # Plot mean path
    mean_path = simulations.mean(axis=0)
    plt.plot(mean_path, color='red', linewidth=2, label='Mean Path')
    
    # Plot percentiles
    percentile_5 = np.percentile(simulations, 5, axis=0)
    percentile_95 = np.percentile(simulations, 95, axis=0)
    plt.plot(percentile_5, color='green', linewidth=2, linestyle='--', label='5th Percentile')
    plt.plot(percentile_95, color='green', linewidth=2, linestyle='--', label='95th Percentile')
    
    plt.axhline(y=initial_investment, color='black', linestyle='--', 
                linewidth=1, label='Initial Investment')
    plt.title('Monte Carlo Portfolio Simulation', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_final_value_distribution(simulations: np.ndarray,
                                 initial_investment: float,
                                 figsize: tuple = (10, 6),
                                 save_path: Optional[str] = None):
    """
    Plot distribution of final portfolio values
    
    Args:
        simulations: Array of simulation paths
        initial_investment: Initial portfolio value
        figsize: Figure size
        save_path: Path to save figure
    """
    final_values = simulations[:, -1]
    
    plt.figure(figsize=figsize)
    plt.hist(final_values, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(x=initial_investment, color='red', linestyle='--', 
                linewidth=2, label='Initial Investment')
    plt.axvline(x=np.mean(final_values), color='green', linestyle='--',
                linewidth=2, label=f'Mean: ${np.mean(final_values):,.0f}')
    plt.axvline(x=np.median(final_values), color='orange', linestyle='--',
                linewidth=2, label=f'Median: ${np.median(final_values):,.0f}')
    
    plt.title('Distribution of Final Portfolio Values', fontsize=14, fontweight='bold')
    plt.xlabel('Portfolio Value ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_efficient_frontier(frontier_df: pd.DataFrame,
                           optimal_portfolio: dict,
                           min_var_portfolio: dict,
                           equal_weight_portfolio: dict,
                           figsize: tuple = (10, 6),
                           save_path: Optional[str] = None):
    """
    Plot efficient frontier with optimal portfolios
    
    Args:
        frontier_df: DataFrame with efficient frontier points
        optimal_portfolio: Optimal (max Sharpe) portfolio metrics
        min_var_portfolio: Minimum variance portfolio metrics
        equal_weight_portfolio: Equal weight portfolio metrics
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    # Plot efficient frontier
    plt.plot(frontier_df['volatility'], frontier_df['return'], 
             'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot special portfolios
    plt.scatter(optimal_portfolio['volatility'], optimal_portfolio['expected_return'],
               marker='*', s=500, c='gold', edgecolors='black', linewidths=2,
               label=f"Max Sharpe (SR={optimal_portfolio['sharpe_ratio']:.2f})", zorder=5)
    
    plt.scatter(min_var_portfolio['volatility'], min_var_portfolio['expected_return'],
               marker='s', s=200, c='red', edgecolors='black', linewidths=2,
               label='Minimum Variance', zorder=5)
    
    plt.scatter(equal_weight_portfolio['volatility'], equal_weight_portfolio['expected_return'],
               marker='o', s=200, c='green', edgecolors='black', linewidths=2,
               label='Equal Weight', zorder=5)
    
    plt.title('Efficient Frontier & Optimal Portfolios', fontsize=14, fontweight='bold')
    plt.xlabel('Volatility (Risk)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_portfolio_weights(weights: dict, title: str = "Portfolio Weights",
                          figsize: tuple = (10, 6),
                          save_path: Optional[str] = None):
    """
    Plot portfolio weight allocation
    
    Args:
        weights: Dictionary of asset weights
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    assets = list(weights.keys())
    weight_values = list(weights.values())
    
    plt.figure(figsize=figsize)
    colors = plt.cm.Set3(range(len(assets)))
    plt.bar(assets, weight_values, color=colors, edgecolor='black', linewidth=1.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Asset', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.ylim(0, max(weight_values) * 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(weight_values):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sensitivity_analysis(sensitivity_df: pd.DataFrame,
                             x_col: str, y_col: str,
                             title: str,
                             figsize: tuple = (10, 6),
                             save_path: Optional[str] = None):
    """
    Plot sensitivity analysis results
    
    Args:
        sensitivity_df: DataFrame with sensitivity results
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    plt.plot(sensitivity_df[x_col], sensitivity_df[y_col], 
             marker='o', linewidth=2, markersize=6)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
