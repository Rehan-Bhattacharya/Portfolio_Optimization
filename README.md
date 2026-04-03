# Portfolio Optimization — Markowitz Mean-Variance Framework
Python implementation of Modern Portfolio Theory applied to 6 NSE-listed Indian equities.

## Features

- Fetches 5 years of adjusted closing prices (2021–2025) via yfinance and computes daily log returns, annualized mean returns,
  volatility, and correlation matrix
- Matrix-based portfolio math engine — return (W^T · R), variance (W^T · Σ · W), and Sharpe ratio — reused across all phases
- Monte Carlo simulation generating 10,000 random weight combinations to map the full opportunity set
- Efficient Frontier traced via scipy.optimize.minimize (SLSQP) across 50 target return levels
- Identifies the Maximum Sharpe (Tangency) Portfolio and Global Minimum Variance Portfolio
- Interactive Plotly chart overlaying the Monte Carlo cloud, Efficient Frontier curve, and both optimal portfolio markers with hover
  tooltips

## Project Structure
- portfolio_optimization.py — single-file, 6-phase implementation

## Libraries Required
pip install numpy pandas scipy plotly yfinance

## How to Get Data
No API key or data files needed. yfinance pulls adjusted closing prices directly at runtime. To change the stock universe, edit the config at the top of the file:

TICKERS  = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS', 'WIPRO.NS']
START_DATE     = '2021-01-01'
END_DATE       = '2025-12-31'
RISK_FREE_RATE = 0.06932

Append .NS for NSE-listed stocks. Any Yahoo Finance ticker works.

## Key Results
**Universe — Annualized Statistics (2021–2025)**

| Stock | Return | Volatility |
|---|---|---|
| ITC.NS | +17.81% | 19.90% |
| RELIANCE.NS | +10.94% | 22.70% |
| HDFCBANK.NS | +7.84% | 21.24% |
| INFY.NS | +7.75% | 24.11% |
| WIPRO.NS | +7.40% | 25.87% |
| TCS.NS | +4.41% | 20.74% |

The IT trio (INFY, TCS, WIPRO) showed pairwise correlations of 0.606–0.698 — high enough that overweighting them together actively increases portfolio variance. ITC showed correlations of 0.177–0.226 with the IT names, making it the dominant diversifier in every optimal portfolio the optimizer found.

Monte Carlo (10,000 simulations): Returns ranged 6.22%–14.27%, volatility 14.01%–20.45%, Sharpe -0.041–0.464. The best random draw landed at Sharpe 0.464 with 57% ITC and 28% RELIANCE — the simulation independently discovered the same concentration the scipy optimizer later confirmed mathematically.

**Optimal Portfolios**

| Portfolio | Return | Volatility | Sharpe | Key Weights |
|---|---|---|---|---|
| Max Sharpe | 17.74% | 19.78% | 0.5465 | ITC 99%, RELIANCE 1% |
| Min Variance | 10.58% | 13.94% | 0.2616 | TCS 27%, ITC 33%, HDFCBANK 24% |

The Max Sharpe result is a corner solution — ITC's combination of highest return and lowest correlation with all other assets causes the unconstrained optimizer to concentrate entirely in it. In a real portfolio context a 40% single-stock cap would be applied. The Min Variance portfolio eliminated WIPRO entirely (highest vol at 25.87%) and nearly eliminated INFY, achieving 13.94% volatility against individual stock volatilities of 19.9%–25.87% — a meaningful reduction from correlation structure alone.

Optimization convergence across methods — Equal Weight Sharpe 0.160 → Monte Carlo best 0.464 → scipy optimum 0.5465 — quantifies what mathematical optimization adds over naive and random approaches.

**Sample Image — Efficient Frontier: Monte Carlo Cloud + Optimal Portfolios**
<img width="992" height="693" alt="Efficient Frontier — Monte Carlo Cloud + Optimal Portfolios" src="https://github.com/user-attachments/assets/5ad90a99-5213-4ead-b5e8-22f7db22ff45" />

