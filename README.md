# Quantitative Financial Analysis Project

A comprehensive Python framework for quantitative financial analysis, portfolio optimization, and risk management. This project focuses on the "Magnificent Seven" tech stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META) and applies advanced computational techniques to support informed investment decision-making.

## Features

- **Data Collection**: Automated stock market data fetching using Yahoo Finance API
- **Monte Carlo Simulation**: Run thousands of portfolio simulations to forecast potential outcomes
- **Portfolio Optimization**: Find optimal asset allocations using Modern Portfolio Theory
  - Maximum Sharpe Ratio optimization
  - Minimum Variance optimization
  - Efficient Frontier generation
- **Risk Analysis**: Comprehensive risk metrics including VaR, CVaR, and volatility analysis
- **Sensitivity Analysis**: Test portfolio robustness across different parameters
- **Visualization**: Professional charts and graphs for all analyses

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

## Quick Start

### Demo Mode (No Internet Required)

Run with synthetic data for testing:
```bash
python demo_analysis.py
```

### Full Analysis (Requires Internet)

Run complete analysis with real market data:
```bash
python main_analysis.py
```

### Simple Example

Quick demonstration with 3 stocks:
```bash
python examples/simple_example.py
```

## Project Structure

```
simulation-project/
├── src/
│   ├── data/              # Data collection modules
│   ├── analysis/          # Analysis algorithms (Monte Carlo, optimization, sensitivity)
│   └── visualization/     # Plotting and charting functions
├── examples/              # Example scripts
├── output/               # Generated visualizations and results
├── main_analysis.py      # Main analysis script (real data)
├── demo_analysis.py      # Demo script (synthetic data)
├── requirements.txt      # Python dependencies
└── DOCUMENTATION.md      # Detailed documentation

```

## Key Outputs

The analysis generates the following visualizations:

1. **price_history.png** - Historical stock price trends
2. **returns_distribution.png** - Return distribution histograms
3. **correlation_heatmap.png** - Asset correlation matrix
4. **efficient_frontier.png** - Risk-return efficient frontier
5. **optimal_weights.png** - Optimal portfolio allocation
6. **monte_carlo_simulation.png** - 10,000 simulated portfolio paths
7. **final_value_distribution.png** - Distribution of final portfolio values
8. **rate_sensitivity.png** - Risk-free rate sensitivity analysis
9. **weight_sensitivity.png** - Portfolio weight sensitivity

## Methodology

### 1. Portfolio Optimization
Uses Modern Portfolio Theory to find optimal asset allocations that:
- Maximize risk-adjusted returns (Sharpe ratio)
- Minimize portfolio volatility
- Balance diversification across assets

### 2. Monte Carlo Simulation
Simulates 10,000 potential portfolio paths over one year to:
- Forecast expected returns and risks
- Calculate probability distributions
- Estimate Value at Risk (VaR) and Conditional VaR

### 3. Risk Analysis
Comprehensive risk metrics including:
- **Sharpe Ratio**: Risk-adjusted return measure
- **VaR (95%)**: Maximum expected loss at 95% confidence
- **CVaR (95%)**: Expected loss in worst 5% of scenarios
- **Volatility**: Standard deviation of returns

### 4. Sensitivity Analysis
Tests portfolio robustness by varying:
- Risk-free interest rates
- Individual asset weights
- Correlation structures

## Example Output

```
Maximum Sharpe Ratio Portfolio:
  Expected Return: 54.33%
  Volatility: 19.83%
  Sharpe Ratio: 2.6390

Monte Carlo Simulation (1-year forecast):
  Expected Portfolio Value: $17,173.48
  Expected Return: 71.73%
  Probability of Loss: 0.37%
  Value at Risk (95%): 21.26%
```

## Documentation

For detailed documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)

## Requirements

- Python 3.8+
- numpy
- pandas
- yfinance
- matplotlib
- scipy
- seaborn

## Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

## License

This project is provided as-is for educational purposes.
