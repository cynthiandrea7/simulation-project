# Quantitative Financial Analysis Framework

## Overview

This project provides a comprehensive framework for quantitative financial analysis, focusing on portfolio optimization, risk management, and investment decision-making using advanced computational techniques.

## Features

### 1. Data Collection
- **Stock Data Collection**: Automated fetching of historical stock market data using Yahoo Finance API
- **Focus on Magnificent Seven**: Pre-configured for analyzing top tech stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META)
- **Flexible Time Periods**: Support for custom date ranges and periods
- **Statistical Analysis**: Built-in calculation of returns, correlations, and summary statistics

### 2. Portfolio Optimization
- **Maximum Sharpe Ratio**: Find the portfolio with the best risk-adjusted returns
- **Minimum Variance**: Identify the least risky portfolio allocation
- **Efficient Frontier**: Generate and visualize the efficient frontier of optimal portfolios
- **Custom Constraints**: Support for custom weight constraints and optimization parameters

### 3. Monte Carlo Simulation
- **Portfolio Forecasting**: Simulate thousands of potential portfolio paths
- **Statistical Analysis**: Calculate mean, median, percentiles, and distributions
- **Risk Metrics**: Compute Value at Risk (VaR) and Conditional VaR (CVaR)
- **Probability Analysis**: Estimate probability of achieving specific returns or losses

### 4. Risk Analysis
- **Volatility Analysis**: Measure and compare portfolio risk
- **Correlation Analysis**: Understand relationships between assets
- **Diversification Metrics**: Evaluate portfolio diversification
- **Value at Risk (VaR)**: Quantify maximum expected loss at given confidence levels
- **Conditional VaR (CVaR)**: Assess tail risk and extreme scenarios

### 5. Sensitivity Analysis
- **Risk-Free Rate Sensitivity**: Analyze impact of interest rate changes
- **Weight Sensitivity**: Examine effects of changing individual asset allocations
- **Correlation Sensitivity**: Test portfolio robustness to correlation changes
- **Monte Carlo Weight Analysis**: Random sampling of portfolio combinations

### 6. Visualization
- **Price History Charts**: Track historical stock prices
- **Return Distributions**: Visualize return characteristics
- **Correlation Heatmaps**: Display relationships between assets
- **Monte Carlo Paths**: Plot simulation trajectories
- **Efficient Frontier**: Visualize risk-return tradeoffs
- **Portfolio Weights**: Display optimal allocations
- **Sensitivity Plots**: Show parameter impact on portfolio metrics

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/cynthiandrea7/simulation-project.git
cd simulation-project
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete analysis on the Magnificent Seven stocks:

```bash
python main_analysis.py
```

This will:
1. Fetch 2 years of historical data
2. Calculate optimal portfolio allocations
3. Run 10,000 Monte Carlo simulations
4. Perform sensitivity analysis
5. Generate visualizations in the `output/` directory

### Simple Example

For a quick demonstration with fewer stocks:

```bash
python examples/simple_example.py
```

### Custom Analysis

```python
from src.data.stock_data import StockDataCollector
from src.analysis.portfolio_optimization import PortfolioOptimizer
from src.analysis.monte_carlo import MonteCarloSimulator

# 1. Collect data for specific stocks
collector = StockDataCollector(['AAPL', 'MSFT', 'GOOGL'])
data = collector.fetch_data(period='2y')
returns = collector.calculate_returns()

# 2. Optimize portfolio
optimizer = PortfolioOptimizer(returns, risk_free_rate=0.02)
optimal_portfolio = optimizer.optimize_sharpe_ratio()

# 3. Run Monte Carlo simulation
mc = MonteCarloSimulator(returns, initial_investment=10000)
weights = [optimal_portfolio['weights'][ticker] for ticker in returns.columns]
simulations = mc.simulate_portfolio(weights, num_simulations=10000, num_days=252)
stats = mc.get_simulation_statistics(simulations)
```

## Project Structure

```
simulation-project/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── stock_data.py           # Data collection and preprocessing
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── monte_carlo.py          # Monte Carlo simulation engine
│   │   ├── portfolio_optimization.py  # Portfolio optimization algorithms
│   │   └── sensitivity_analysis.py # Sensitivity analysis tools
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                # Plotting and visualization functions
├── examples/
│   └── simple_example.py           # Simple usage example
├── output/                         # Generated plots and results
├── main_analysis.py                # Main analysis script
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
└── DOCUMENTATION.md                # This file
```

## Core Concepts

### Modern Portfolio Theory (MPT)
The framework implements Markowitz's Modern Portfolio Theory to find optimal asset allocations that maximize returns for a given level of risk.

### Sharpe Ratio
The Sharpe ratio measures risk-adjusted returns:
```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
```
Higher Sharpe ratios indicate better risk-adjusted performance.

### Monte Carlo Simulation
Monte Carlo methods simulate thousands of possible future scenarios based on historical return characteristics, providing probability distributions of potential outcomes.

### Value at Risk (VaR)
VaR estimates the maximum loss expected over a given time period at a specified confidence level. For example, a 95% VaR of -5% means there's a 95% probability that losses won't exceed 5%.

### Conditional VaR (CVaR)
Also called Expected Shortfall, CVaR measures the expected loss in the worst-case scenarios beyond the VaR threshold, providing a more conservative risk measure.

## Outputs

The main analysis generates the following visualizations in the `output/` directory:

1. **price_history.png**: Historical stock prices
2. **returns_distribution.png**: Distribution of daily returns for each stock
3. **correlation_heatmap.png**: Correlation matrix between stocks
4. **efficient_frontier.png**: Efficient frontier with optimal portfolios
5. **optimal_weights.png**: Optimal portfolio allocation (max Sharpe)
6. **equal_weights.png**: Equal-weighted portfolio allocation
7. **monte_carlo_simulation.png**: Simulation paths over time
8. **final_value_distribution.png**: Distribution of final portfolio values
9. **rate_sensitivity.png**: Sensitivity to risk-free rate changes
10. **weight_sensitivity.png**: Sensitivity to portfolio weight changes

## Key Metrics Explained

### Expected Return
The mean annualized return of the portfolio based on historical data.

### Volatility (Standard Deviation)
A measure of portfolio risk - higher volatility means greater uncertainty in returns.

### Sharpe Ratio
Risk-adjusted return metric - higher is better. Typical values:
- < 1.0: Poor risk-adjusted returns
- 1.0-2.0: Good risk-adjusted returns
- > 2.0: Excellent risk-adjusted returns

### Correlation
Measures how stocks move together (-1 to 1):
- -1: Perfect negative correlation
- 0: No correlation
- 1: Perfect positive correlation

Lower correlations between assets improve diversification benefits.

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **yfinance**: Yahoo Finance API for stock data
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing and optimization
- **seaborn**: Statistical data visualization

## Best Practices

1. **Data Quality**: Use sufficient historical data (at least 1-2 years) for reliable statistics
2. **Rebalancing**: Consider periodic portfolio rebalancing to maintain target allocations
3. **Risk Tolerance**: Adjust optimization constraints based on individual risk tolerance
4. **Diversification**: Don't over-concentrate in single assets or sectors
5. **Regular Updates**: Refresh analysis periodically as market conditions change
6. **Backtesting**: Validate strategies on out-of-sample data before implementation

## Limitations

- Historical data may not predict future performance
- Assumes returns follow normal distribution (may not hold during market stress)
- Transaction costs and taxes not included in optimization
- No consideration of market impact or liquidity constraints
- Optimization based on historical correlations which may change

## Future Enhancements

Potential additions to the framework:
- Real-time data streaming
- Machine learning for return prediction
- Factor models (Fama-French, etc.)
- Options and derivatives pricing
- Backtesting engine with transaction costs
- Risk parity and other allocation strategies
- Regime detection and adaptive allocation
- ESG (Environmental, Social, Governance) screening

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This software is for educational purposes only. It should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.
