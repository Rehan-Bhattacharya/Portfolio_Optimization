
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# Our Portfolio Universe: 
TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS', 'WIPRO.NS']

START_DATE = '2021-01-01'
END_DATE   = '2025-12-31'

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06932               # India 10 year Goverment Bond Yield 


# Phase1 : Fetching and processing stock data 

def fetch_stock_data (tickers, start, end):
    """
    Downloading adjusted closing price for all tickers
    Rows = Date , Column = tickers
    """

    print(f"\n{'='*55}")
    print(f"  Fetching price data for: {tickers}")
    print(f"  Period: {start} to {end}")
    print(f"{'='*55}")

    raw = yf.download(tickers , start=start, end=end, auto_adjust=True, progress=False)
    prices = raw['Close']
    prices.dropna(how= 'all', inplace=True)                     # dropping any rows with NaN values

    print(f"\n Downloaded {len(prices)} trading days of data")
    print(f"  Date range: {prices.index[0].date()} - {prices.index[-1].date()}")
    print(f"\nFirst 5 rows of price data:")
    print(prices.head().round(2))

    return prices

def compute_returns(prices):
    """
    Computing daily log returns from price data 
    log return = ln(P_t / P_(t-1))
    """

    log_returns = np.log(prices / prices.shift(1))
    log_returns.dropna(inplace = True)

    print(f"\n{'='*55}")
    print(f"  Daily Log Returns")
    print(f"{'='*55}")
    print(f"\nFirst 5 rows of log returns:")
    print(log_returns.head().round(5))

    return log_returns

def compute_annual_stats(log_returns):
    """
    Annualizing mean returns and covariance matrix
    Mean return   * 252
    Covariance    * 252   (variance scales linearly with time)
    """
    
    annual_returns = log_returns.mean() * TRADING_DAYS
    annual_cov     = log_returns.cov()  * TRADING_DAYS
    correlation    = log_returns.corr()

    print(f"\n{'='*55}")
    print(f"  Annualized Statistics")
    print(f"{'='*55}")

    print("\nAnnualized Mean Returns:")
    for ticker, ret in annual_returns.items():
        print(f"  {ticker:6s}: {ret*100:+.2f}%")
    
    print("\nAnnualized Volatility (Std Dev):")
    for ticker in annual_returns.index:
        vol = np.sqrt(annual_cov.loc[ticker, ticker])
        print(f"  {ticker:6s}: {vol*100:.2f}%")

    print("\nCorrelation Matrix:")
    print(correlation.round(3))

    return annual_returns, annual_cov, correlation


# Phase2 : Portfolio Maths - Weights, Variance and Sharpe Ratio

def portfolio_performance(weights, annual_returns, annual_cov):
    """
    Using an array of weights, we compute the following
    - Portfolio expected return
    - Portfolio standard deviation (volatility)
    - Sharpe ratio

    Parameters:
    weights        : numpy array 
    annual_returns : pandas Series  — annualised mean returns
    annual_cov     : pandas DataFrame — annualised covariance matrix
    """

    # Calculating portfolio return 
    port_return = float(np.dot(weights, annual_returns))

    # Calculating portfolio variance and volatility
    port_variance   = float(np.dot(weights.T , np.dot(annual_cov, weights)))
    port_volatility = float(np.sqrt(port_variance))

    # Sharpe Ratio
    sharpe_ratio = float((port_return - RISK_FREE_RATE) / port_volatility)

    return port_return, port_volatility, sharpe_ratio

def test_portfolios(annual_returns , annual_cov):
    """
    Testing three specific portfolios to verify the math which we have build above:
    1. Equal weight    — same allocation to every stock
    2. ITC heavy       — overweight the best performer
    3. IT heavy        — overweight the correlated IT stocks
    
    This helps build intuition before Monte Carlo randomises weights.
    """

    n = len(annual_returns)                # number of stocks = 6 

    # Portfolio 1: Equal Weight
    equal_weights = np.array([1/n] * n) 

    # Portfolio 2: ITC Heavy () - as ITC had the highest return (17.81%) and lowest volitility (19.90%)
    
    # Allocate 40% to ITC, spread rest equally across others
    itc_heavy = np.array([0.10, 0.10, 0.50, 0.10, 0.10, 0.10])

    # Portfolio 3: IT Heavy - Overweighing INFY, TCS, WIPRO — highly correlated IT trio
    it_heavy = np.array([0.10, 0.30, 0.05, 0.10, 0.30, 0.15])

    portfolios = {
        'Equal Weight'  : equal_weights,
        'ITC Heavy'     : itc_heavy,
        'IT Heavy'      : it_heavy
    }

    print(f"\n{'='*60}")
    print(f"  Phase 2 — Portfolio Performance Comparison")
    print(f"{'='*60}")
    print(f"\n{'Portfolio':<20} {'Return':>10} {'Volatility':>12} {'Sharpe':>10}")
    print(f"{'-'*55}")

    for name, weights in portfolios.items():
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights don't sum to 1 for {name}!"

        ret, vol, sharpe = portfolio_performance(weights, annual_returns, annual_cov)

        print(f"{name:<20} {ret*100:>9.2f}%  {vol*100:>10.2f}%  {sharpe:>10.4f}")

    print(f"\n{'='*60}")
    print(f"  Weight Breakdown")
    print(f"{'='*60}")

    for name, weights in portfolios.items():
        print(f"\n{name}:")
        for ticker, w in zip(annual_returns.index, weights):
            bar = '█' * int(w * 40)                               # simple visual bar
            print(f"  {ticker:<15} {w*100:>5.1f}%  {bar}")


# Phase3: Monte Carlo Simulation - 10,000 Random Portfolios

def run_monte_carlo_simulation(annual_returns, annual_cov, n_portfolios = 10000):
    """
    Generate n_portfolios random weight combinations and compute
    performance for each one.

    Returns a dictionary of arrays — one value per portfolio:
        returns     : array of portfolio returns
        volatility  : array of portfolio volatilities
        sharpe      : array of sharpe ratios
        weights     : 2D array of weights (n_portfolios X n_stocks)
    """

    n_stocks = len(annual_returns)

    # Pre allocating empty arrays 
    mc_returns = np.zeros(n_portfolios)
    mc_volatility = np.zeros(n_portfolios)
    mc_sharpe     = np.zeros(n_portfolios)
    mc_weights    = np.zeros((n_portfolios, n_stocks))   # 2D: each row = one portfolio's weights

    print(f"\n{'='*55}")
    print(f"  Phase 3 — Monte Carlo Simulation")
    print(f"  Running {n_portfolios:,} random portfolios...")
    print(f"{'='*55}")

    # Main Monte Carlo Simulation loop
    for i in range(n_portfolios):
        raw_weights = np.random.random(n_stocks)                                           #generating random weights for 6 stocks that sum to 1 
        weights     = raw_weights / raw_weights.sum()

        ret, vol, sharpe = portfolio_performance(weights, annual_returns, annual_cov)      # computing performance using phase 2 function 

        mc_returns[i]      = ret                                                           # storing results in index i
        mc_volatility[i]   = vol
        mc_sharpe[i]       = sharpe
        mc_weights[i, :]   = weights 
        
    
    # Summary Statistics
    print(f"\n  Simulation Complete!")
    print(f"\n  Return   — Min: {mc_returns.min()*100:.2f}%  "
          f"Max: {mc_returns.max()*100:.2f}%")
    print(f"  Volatility— Min: {mc_volatility.min()*100:.2f}%  "
          f"Max: {mc_volatility.max()*100:.2f}%")
    print(f"  Sharpe   — Min: {mc_sharpe.min():.4f}  "
          f"Max: {mc_sharpe.max():.4f}")
    
    # Best portfolio from simulation 
    best_idx = np.argmax(mc_sharpe)

    print(f"\n  Best Sharpe from Monte Carlo: {mc_sharpe[best_idx]:.4f}")
    print(f"  Return:     {mc_returns[best_idx]*100:.2f}%")
    print(f"  Volatility: {mc_volatility[best_idx]*100:.2f}%")
    print(f"\n  Weight Breakdown (Best MC Portfolio):")
    for ticker, w in zip(annual_returns.index, mc_weights[best_idx]):
        bar = '█' * int(w * 40)
        print(f"    {ticker:<15} {w*100:>5.1f}%  {bar}")

    results = {
        'returns'    : mc_returns,
        'volatility' : mc_volatility,
        'sharpe'     : mc_sharpe,
        'weights'    : mc_weights
    }

    return results


# Phase4 : Efficient Frontier - scipy optimization

def minimise_volatility(weights, annual_returns, annual_cov):
    """
    This is the objective function for scipy — returns portfolio volatility.
    scipy.minimize will try to make this number as small as possible by adjusting the weights.
    """
    _, vol, _ = portfolio_performance(weights, annual_returns, annual_cov)
    return vol


def get_efficient_frontier(annual_returns, annual_cov, n_points=50):
    """
    Computing the efficient frontier by minimising volatility for each target return level between min and max possible return.
    Parameters:
        annual_returns : pandas Series of annualised returns
        annual_cov     : pandas DataFrame of annualised covariance
        n_points       : number of points on the frontier (default 50)
    """

    n_stocks = len(annual_returns)

    # Bounds - creating the bound for weights - each weight must be between 0 and 1
    bounds = tuple((0, 1) for _ in range(n_stocks))                                       # Bounds only control each weight individually - They have no idea what the otehr weights are doing

    # Base constraint - all the weights must sum to 1 
      # weights.sum() - 1 = 0  i.e  weights.sum() = 1 
    sum_constraint = {
        'type' : 'eq',
        'fun'  : lambda w : np.sum(w) - 1 
    }

    # Target return range 
    max_return = annual_returns.max()
    min_return = annual_returns.min()
    target_returns = np.linspace(min_return, max_return , n_points)

    # Pre-allocating space for the result arrays 
    frontier_returns    = np.zeros(n_points)
    frontier_volatility = np.zeros(n_points)
    frontier_sharpe     = np.zeros(n_points)
    frontier_weights    = np.zeros((n_points, n_stocks))

    # Starting with equal weights 
      # as scipy needs a starting point for its search
    initial_weights = np.array([1/ n_stocks] * n_stocks)

    print(f"\n{'='*55}")
    print(f"  Phase 4 — Efficient Frontier")
    print(f"  Optimising {n_points} portfolios along the frontier...")
    print(f"{'='*55}")

    # Main optimization loop - for every target return level - finding weights whihc result in minimum volatility
    for i , target_ret in enumerate(target_returns):
        # creating a second constraint - return constraint 
        return_constraint = {
            'type' : 'eq',
            'fun'  : lambda w , tr = target_ret : portfolio_performance( w , annual_returns, annual_cov)[0] - tr
        }

        constraints = [sum_constraint , return_constraint]        

        result = minimize(                         # running scipy optimization
            fun    = minimise_volatility,          
            x0     = initial_weights,              
            args   = (annual_returns, annual_cov), 
            method = 'SLSQP',                      # Sequential Least Squares — best for portfolio optimisation
            bounds = bounds,                       
            constraints = constraints 
        )

        # storing results if optimization successful
        if result.success:
            opt_weights = result.x                 # the optimal weights scipy found
            ret, vol, sharpe = portfolio_performance(
                opt_weights, annual_returns, annual_cov
            )
            frontier_returns[i]    = ret
            frontier_volatility[i] = vol
            frontier_sharpe[i]     = sharpe
            frontier_weights[i, :] = opt_weights

    print(f"\n  Frontier computed successfully!")
    print(f"\n  {'Target Return':<20} {'Volatility':>12} {'Sharpe':>10}")
    print(f"  {'-'*45}")

    # Printing every 5th point to avoid flooding the terminal
    for i in range(0, n_points, 5):
        print(f"  {frontier_returns[i]*100:>10.2f}%          "
              f"{frontier_volatility[i]*100:>8.2f}%   "
              f"{frontier_sharpe[i]:>10.4f}")

    return frontier_returns, frontier_volatility, frontier_sharpe, frontier_weights 

# Phase5 : Optimal Portfolios - Minimizing Variance & Maximizing Sharpe Ratio

def maximize_sharpe (weights, annual_returns, annual_cov):
    """
    This is an objective function for maximum Sharpe optimization
    This function returns negative Sharpe- since scipy can only minimize , so minimizing negative Sharpe = maximizing Sharpe
    """

    _,vol,sharpe = portfolio_performance(weights, annual_returns, annual_cov)
    return -sharpe

def optimal_portfolios(annual_returns, annual_cov):
    """
    Here we are finding two optimal portfolios
    1. Maximum Sharpe Ratio
    2. Minimum Variance 
    
    Both uses scipy.minimize but with different objective goals
    """

    n_stocks = len(annual_returns)

    # bounds and sum constraint (same as phase 4)

    bounds = tuple((0,1) for _ in range(n_stocks))

    sum_constraint = {
        'type' : 'eq',
        'fun'  : lambda w : np.sum(w) - 1
    }

    constraints = [sum_constraint]                          # only one constraint - no return constraint
    initial_weight = np.array([1 / n_stocks] * n_stocks)    # we need to give scipy an initial point to start

    print(f"\n{'='*55}")
    print(f"  Phase 5 — Optimal Portfolios")
    print(f"{'='*55}")

    # Optimization 1 : maximum Sharpe
    print("\n  Finding Maximum Sharpe Portfolio...")

    max_sharpe_ratio = minimize(
        fun         = maximize_sharpe,
        x0          = initial_weight,
        args        = (annual_returns, annual_cov),
        method      = 'SLSQP',
        bounds      = bounds,
        constraints = constraints
    )
    max_sharpe_weights = max_sharpe_ratio.x
    max_sharpe_return = portfolio_performance( max_sharpe_weights, annual_returns, annual_cov)

    # Optimization 2 : minimum variance
    print(f"  Finding Minimum Variance Portfolio...")

    min_var_result = minimize(
        fun         = minimise_volatility,         # reusing Phase 4 function
        x0          = initial_weight,
        args        = (annual_returns, annual_cov),
        method      = 'SLSQP',
        bounds      = bounds,
        constraints = constraints                  
    )
    min_var_weights = min_var_result.x
    min_var_perf    = portfolio_performance(min_var_weights, annual_returns, annual_cov)

    # Printing results
    print(f"\n{'='*55}")
    print(f"  MAXIMUM SHARPE PORTFOLIO (Tangency Portfolio)")
    print(f"{'='*55}")
    print(f"  Return:     {max_sharpe_return[0]*100:.2f}%")
    print(f"  Volatility: {max_sharpe_return[1]*100:.2f}%")
    print(f"  Sharpe:     {max_sharpe_return[2]:.4f}")
    print(f"\n  Weight Breakdown:")
    for ticker, w in zip(annual_returns.index, max_sharpe_weights):
        bar = '█' * int(w * 40)
        print(f"    {ticker:<15} {w*100:>5.1f}%  {bar}")

    print(f"\n{'='*55}")
    print(f"  MINIMUM VARIANCE PORTFOLIO")
    print(f"{'='*55}")
    print(f"  Return:     {min_var_perf[0]*100:.2f}%")
    print(f"  Volatility: {min_var_perf[1]*100:.2f}%")
    print(f"  Sharpe:     {min_var_perf[2]:.4f}")
    print(f"\n  Weight Breakdown:")
    for ticker, w in zip(annual_returns.index, min_var_weights):
        bar = '█' * int(w * 40)
        print(f"    {ticker:<15} {w*100:>5.1f}%  {bar}")
    
    # Head to head comparison
    print(f"\n{'='*55}")
    print(f"  HEAD TO HEAD COMPARISON")
    print(f"{'='*55}")
    print(f"\n  {'Metric':<20} {'Max Sharpe':>12} {'Min Variance':>14}")
    print(f"  {'-'*48}")
    print(f"  {'Return':<20} {max_sharpe_return[0]*100:>11.2f}%  {min_var_perf[0]*100:>12.2f}%")
    print(f"  {'Volatility':<20} {max_sharpe_return[1]*100:>11.2f}%  {min_var_perf[1]*100:>12.2f}%")
    print(f"  {'Sharpe':<20} {max_sharpe_return[2]:>12.4f}  {min_var_perf[2]:>12.4f}")

    return max_sharpe_weights, min_var_weights, max_sharpe_return, min_var_perf


# Phase 6: Visualizations

def plot_phase1_charts(prices, log_returns, correlation):
    """
    Here we are creating three charts from phase 1 data :
    1. Normalized price performance
    2. Daily return distributions
    3. Correlation heatmap
    """
    # Chart 1: Normalized Price Performance 
    fig1 = go.Figure()
    normalized = (prices / prices.iloc[0]) * 100   # rebase to 100

    for col in normalized.columns:
        fig1.add_trace(go.Scatter(
            x    = normalized.index,
            y    = normalized[col],
            name = col,
            mode = 'lines'
        ))

    fig1.update_layout(
        title      = 'Chart 1 — Normalized Price Performance (Base = 100)',
        xaxis_title= 'Date',
        yaxis_title= 'Indexed Price (Start = 100)',
        template   = 'plotly_dark',
        hovermode  = 'x unified'
    )
    fig1.show()

    # Chart 2: Return Distributions
    fig2 = go.Figure()

    for col in log_returns.columns:
        fig2.add_trace(go.Histogram(
            x       = log_returns[col],
            name    = col,
            opacity = 0.6,
            nbinsx  = 80
        ))

    fig2.update_layout(
        title      = 'Chart 2 — Daily Log Return Distributions',
        xaxis_title= 'Daily Log Return',
        yaxis_title= 'Frequency',
        template   = 'plotly_dark',
        barmode    = 'overlay'
    )
    fig2.show()

    # Chart 3: Correlation Heatmap
    fig3 = go.Figure(data=go.Heatmap(
        z             = correlation.values,
        x             = correlation.columns.tolist(),
        y             = correlation.index.tolist(),
        colorscale    = 'RdBu_r',
        zmin          = -1,
        zmax          =  1,
        text          = correlation.round(2).values,
        texttemplate  = '%{text}',
        colorbar      = dict(title='Correlation')
    ))

    fig3.update_layout(
        title    = 'Chart 3 — Stock Correlation Matrix',
        template = 'plotly_dark'
    )
    fig3.show()

def plot_efficient_frontier ( mc_results, frontier_returns, frontier_volatility, annual_returns, max_sharpe_return, min_var_perf,max_sharpe_weights, min_var_weights):
    """
    The main chart — layers four elements:
    1. Monte Carlo scatter colored by Sharpe ratio
    2. Efficient Frontier curve
    3. Maximum Sharpe Portfolio marker
    4. Minimum Variance Portfolio marker

    Parameters:
        mc_results         : dictionary from Phase 3
        frontier_returns   : array from Phase 4
        frontier_volatility: array from Phase 4
        annual_returns     : pandas Series from Phase 1
        max_sharpe_perf    : (ret, vol, sharpe) tuple from Phase 5
        min_var_perf       : (ret, vol, sharpe) tuple from Phase 5
        max_sharpe_weights : weights array from Phase 5
        min_var_weights    : weights array from Phase 5
    """

    fig = go.Figure()

    # Layer 1: Monte Carlo Scatter
     # Color = Sharpe ratio (brighter = better risk-adjusted return)
    fig.add_trace(go.Scatter(
        x    = mc_results['volatility'] * 100,
        y    = mc_results['returns']    * 100,
        mode = 'markers',
        name = 'Random Portfolios',
        marker = dict(
            size   = 3,
            color  = mc_results['sharpe'],    # color mapped to Sharpe
            colorscale = 'Viridis',           # dark purple=low, yellow=high
            showscale  = False,
            colorbar   = dict(
                title      = dict(
                    text = 'Sharpe Ratio',
                    side = 'right'
                )
            ),
            opacity = 0.6
        ),
        # Hover text — what shows when you mouse over a dot
        hovertemplate = (
            'Return: %{y:.2f}%<br>'
            'Volatility: %{x:.2f}%<br>'
            'Sharpe: %{marker.color:.4f}<extra></extra>'
        )
    ))

    # Layer 2: Efficient Frontier Curve 
     # The mathematically optimal edge of the cloud
    fig.add_trace(go.Scatter(
        x    = frontier_volatility * 100,
        y    = frontier_returns    * 100,
        mode = 'lines+markers',
        name = 'Efficient Frontier',
        line = dict(
            color = 'white',
            width = 2.5
        ),
        marker = dict(size=5, color='white'),
        hovertemplate = (
            'Frontier Portfolio<br>'
            'Return: %{y:.2f}%<br>'
            'Volatility: %{x:.2f}%<extra></extra>'
        )
    ))

    # Layer 3: Maximum Sharpe Portfolio 
     # Building hover text showing weight breakdown
    max_sharpe_hover = (
        '<b>Maximum Sharpe Portfolio ★</b><br>'
        f'Return: {max_sharpe_return[0]*100:.2f}%<br>'
        f'Volatility: {max_sharpe_return[1]*100:.2f}%<br>'
        f'Sharpe: {max_sharpe_return[2]:.4f}<br>'
        '<br><b>Weights:</b><br>' +
        '<br>'.join([
            f'{ticker}: {w*100:.1f}%'
            for ticker, w in zip(annual_returns.index, max_sharpe_weights)
            if w > 0.001    # only show stocks with meaningful allocation
        ])
    )

    fig.add_trace(go.Scatter(
        x    = [max_sharpe_return[1] * 100],
        y    = [max_sharpe_return[0] * 100],
        mode = 'markers',
        name = 'Max Sharpe Portfolio',
        marker = dict(
            symbol = 'star',
            size   = 20,
            color  = 'gold',
            line   = dict(color='white', width=1)
        ),
        hovertemplate = max_sharpe_hover + '<extra></extra>'
    ))

    # Layer 4: Minimum Variance Portfolio 
    min_var_hover = (
        '<b>Minimum Variance Portfolio ◆</b><br>'
        f'Return: {min_var_perf[0]*100:.2f}%<br>'
        f'Volatility: {min_var_perf[1]*100:.2f}%<br>'
        f'Sharpe: {min_var_perf[2]:.4f}<br>'
        '<br><b>Weights:</b><br>' +
        '<br>'.join([
            f'{ticker}: {w*100:.1f}%'
            for ticker, w in zip(annual_returns.index, min_var_weights)
            if w > 0.001
        ])
    )

    fig.add_trace(go.Scatter(
        x    = [min_var_perf[1] * 100],
        y    = [min_var_perf[0] * 100],
        mode = 'markers',
        name = 'Min Variance Portfolio',
        marker = dict(
            symbol = 'diamond',
            size   = 16,
            color  = 'cyan',
            line   = dict(color='white', width=1)
        ),
        hovertemplate = min_var_hover + '<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title = dict(
            text     = 'Chart 4 — Portfolio Efficient Frontier<br>'
                       '<sup>Indian Equity Portfolio | 2021-2025 | '
                       'Risk Free Rate: 6.932%</sup>',
            font     = dict(size=18)
        ),
        xaxis = dict(
            title      = 'Annual Volatility (Risk) %',
            ticksuffix = '%',
            gridcolor  = 'rgba(255,255,255,0.1)'
        ),
        yaxis = dict(
            title      = 'Annual Return %',
            ticksuffix = '%',
            gridcolor  = 'rgba(255,255,255,0.1)'
        ),
        template    = 'plotly_dark',
        hovermode   = 'closest',
        showlegend  = True,
        legend      = dict(
            x         = 0.02,
            y         = 0.98,
            bgcolor   = 'rgba(0,0,0,0.5)',
            bordercolor = 'white',
            borderwidth = 1
        ),
        width  = 1000,
        height = 700
    )

    fig.show()

    




# Quick test 
def main():
    # Phase 1
    prices         = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    log_returns    = compute_returns(prices)
    annual_returns, annual_cov, correlation = compute_annual_stats(log_returns)
    # Phase 2
    test_portfolios(annual_returns, annual_cov)
    # Phase 3
    mc_results = run_monte_carlo_simulation(annual_returns, annual_cov, n_portfolios=10000)
    # Phase 4 
    frontier_returns, frontier_volatility, frontier_sharpe, frontier_weights = get_efficient_frontier(annual_returns, annual_cov, n_points=50)
    # Phase 5
    max_sharpe_weights, min_var_weights, max_sharpe_return, min_var_perf = optimal_portfolios(annual_returns, annual_cov)
    # Phase 6
       #plot_phase1_charts(prices, log_returns, correlation)

    plot_efficient_frontier(
        mc_results         = mc_results,
        frontier_returns   = frontier_returns,
        frontier_volatility= frontier_volatility,
        annual_returns     = annual_returns,
        max_sharpe_return  = max_sharpe_return,
        min_var_perf       = min_var_perf,
        max_sharpe_weights = max_sharpe_weights,
        min_var_weights    = min_var_weights
    )



if __name__ == '__main__':
    main()


