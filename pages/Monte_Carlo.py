import streamlit as st

# Basic
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from scipy.stats import norm
import pandas as pd
import datetime
import calendar

# Finance
import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg
from nsetools import Nse
from nsepython import nse_get_all_stock_codes

st.title("Monte Carlo Simulation")
st.subheader("Lognormal Mean-Reverting Stochastic Volatility Model for Option Pricing", divider='red')

with st.expander("Model Details"):
    st.markdown(r"""
    This model simulates the evolution of an asset price where the volatility is stochastic and mean-reverting in **log-space**. It captures more realistic market behavior by allowing volatility to vary over time, unlike the constant-volatility Black-Scholes model.

    I model the dynamics using the following **discrete-time stochastic difference equations**:

    #### Asset Price and Volatility Equations
    """)

    st.latex(r'''
    \ln \left( \frac{S_{t+1}}{S_t} \right) = \mu \, \Delta t + \exp(h_t) \sqrt{\Delta t} \cdot \xi_{t+1}
    ''')
    st.latex(r'''
    h_{t+1} = (1 - \rho) \bar{h} + \rho h_t + \nu \sqrt{\Delta t} \cdot \eta_{t+1}
    ''')

    st.markdown(r"""
    - $S_t$ : asset price at time $t$
    - $h_t$: log-volatility at time $t$  
    - $\mu$ : drift term (expected return)  
    - $\bar{h}$ : long-run average of log-volatility  
    - $\rho \in [0, 1]$ : persistence of log-volatility (higher → slower reversion)  
    - $\nu$ : volatility of the log-volatility  
    - $\xi_{t+1}, \eta_{t+1} \sim \mathcal{N}(0, 1)$ : i.i.d. standard Gaussian noise  
    - $\Delta t$ : time step (e.g. 1 trading day = 1/252)

    The asset exhibits **lognormal returns** with **stochastic time-varying volatility**, and the volatility process $h_t$ mean-reverts to $\bar{h}$ over time, ensuring statistical stability.
    """)


#--------------------------------------------------------------------------

# Calculating days till expiry 

def next_monday(date):
    days_ahead = 0 - date.weekday()  # Monday is 0
    if days_ahead <= 0:
        days_ahead += 7
    return date + datetime.timedelta(days=days_ahead)

def last_monday(year, month):
    # Find last day of month
    last_day = datetime.date(year, month, calendar.monthrange(year, month)[1])
    # Backtrack to last Monday
    offset = (last_day.weekday() - 0) % 7  # Monday = 0
    return last_day - datetime.timedelta(days=offset)

def count_weekdays(start_date, end_date):
    day_count = 0
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday=0 ... Friday=4
            day_count += 1
        current += datetime.timedelta(days=1)
    return day_count

# Today
today = datetime.date.today()

# Get expiry dates
weekly_expiry = next_monday(today)
monthly_expiry = last_monday(today.year, today.month)

# Example user choice:
option_expiry = st.sidebar.selectbox('Option expiry :', ('Weekly', 'Monthly'),)

if option_expiry == 'Weekly':
    expiry = weekly_expiry
else:
    expiry = monthly_expiry

n = count_weekdays(today, expiry)

col1, col2 = st.columns(2)

with col1:
    st.date_input("Current Date", value=today, disabled=True)

with col2:
    st.date_input("Expiry Date", value=expiry, disabled=True)
#------------------------------------------------------------------------------

# Fetching realtime parameters from NSE
# nse = Nse()
# all_stock_codes = nse.get_stock_codes()  # returns dict with symbol:name

all_stock_codes = nse_get_all_stock_codes()

# Convert dict keys (symbols) to tuple (excluding first key which is 'SYMBOL')
nse_symbols = tuple(all_stock_codes)[1:]

stock = st.sidebar.selectbox(
    "Select NSE stock",
    nse_symbols,
)

# Define ticker and dates
ticker = f"{stock}.NS"
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=252)  # 1 year ago

# Download data
data = yf.download(ticker, start=start_date, end=end_date)

st.subheader("Latest Week's Data")
st.dataframe(data.tail())

# Compute daily log returns
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# Parameters
S0 = data['Close'].iloc[-1].iloc[0]                # Latest adjusted close price
μ = data['log_return'].mean() * 252                # Annualized mean return

# Calculating rho and nu
log_returns = data['log_return']
window = 20
realized_vol = log_returns.rolling(window).std() * np.sqrt(252)
realized_vol = realized_vol.dropna()
log_vol = np.log(realized_vol)

# AR(1) fit
model = AutoReg(log_vol, lags=1).fit()
ρ = model.params.iloc[1]
ν = np.std(model.resid)
h_bar = log_vol.iloc[-1]                           # log-volatility
h0 = h_bar 
# Option
K = st.sidebar.number_input("Strike Price : ", value=S0)
n = count_weekdays(today, expiry)

risk_free_rate = st.sidebar.slider("Risk free rate (in %) : ", min_value=0, max_value=100, value=6)/100
β = np.exp(-risk_free_rate/252) 

M = 10**st.sidebar.slider("Number of simulations (order of magnitude)", min_value=0, max_value=10, value=6, step=1)

st.subheader("Parameters")
parameter_dict = {
    'Parameters': ['Intial Stock Price (S_0)', 'Drift (μ)', 'Correlation Coefficient (ρ)', 'Volatility of Volatility (ν)', 'Log Volatilty Mean Reversion Rate (h_bar)', 'Initial Log Volatility (h_0)', 'Strike Price (K)', 'Days till expiry (n)', 'Discount Rate (β)'],
    'Values': [S0, μ, ρ, ν, h_bar, h0, K, n, β]
}
df = pd.DataFrame.from_dict(parameter_dict)
st.dataframe(df)

#----------------------------------------------------------------------------------------------------

# Simulate asset price path

h=np.zeros(n+1)

def simulate_asset_price_path(S0=S0, mu=μ, h_bar=h_bar, h0=h0,
                            nu=ν, rho=ρ, n=n):

    dt = 1/252
    s = np.zeros(n + 1)
    h = np.zeros(n + 1)
    s[0] = np.log(S0)  # Working in log space
    h[0] = h0

    for t in range(1, n+1):
        # Generate correlated random shocks
        ξ = np.random.randn()
        η = np.random.randn()
        
        # Update volatility process (YOUR EQUATION)
        h[t] = (1 - rho) * h_bar + rho * h[t-1] + nu * np.sqrt(dt) * η
        
        # Update log-price process (YOUR EQUATION)
        s[t] = s[t-1] + mu * dt + np.exp(h[t-1]) * np.sqrt(dt) * ξ
    
    return np.exp(s)  # Convert back to price space


# Chart asset price paths
st.subheader('Simulated Asset Price Paths')
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Asset Price Paths")

# Simulate and plot 50 paths
for _ in range(50):
    path = simulate_asset_price_path()
    ax.plot(path, alpha=0.5, linewidth=0.8)

# Add current price line (red dashed)
ax.axhline(S0, color='red', linestyle='--', linewidth=1.5, label=f'Current Price: ₹{S0:.2f}')
ax.set_xlabel("Time Steps", fontsize=10)
ax.set_ylabel("Price (₹)", fontsize=10)
ax.grid(True, linestyle=':', alpha=0.3)

# Add parameter box
param_text = (f"Parameters:\n"
              f"μ = {μ}\n"
              f"h_bar = {h_bar}\n"
              f"ν = {ν}\n"
              f"ρ = {ρ}")
ax.text(0.82, 0.75, param_text, transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8), fontsize=9)

# Display the plot in Streamlit
st.pyplot(fig)
#----------------------------------------------------------------------------------------------

# Computing option prices
# Note that in option pricing, we have used mu=risk_free_rate for risk neutral pricing

def compute_option_prices(β=β, μ=risk_free_rate, S0=S0, h0=h0, h_bar=h_bar, K=K, n=n, ρ=ρ, ν=ν, M=M):
# Preallocate arrays for log-price and log-volatility
    s = np.full(M, np.log(S0))
    h = np.full(M, h0)
    dt = 1/252

    # Time loop — but vectorized over all M paths
    for _ in range(n):
        # Generate M normal random variables for both noise terms
        ξ = np.random.randn(M)
        η = np.random.randn(M)

        # Update s and h in-place
        s += μ * dt + np.exp(h) * np.sqrt(dt) * ξ
        h= (1-ρ) * h_bar + ρ * h + ν * np.sqrt(dt)* η

    # Final asset prices
    S_n = np.exp(s)

    # Payoffs
    call_payoffs = np.maximum(S_n - K, 0)
    put_payoffs = np.maximum(K - S_n, 0)

    # Discounted prices and standard errors
    discount = β ** n
    call_price = discount * call_payoffs.mean()
    put_price = discount * put_payoffs.mean()
    
    call_std_err = discount * call_payoffs.std(ddof=1) / np.sqrt(M)
    put_std_err = discount * put_payoffs.std(ddof=1) / np.sqrt(M)

    return (call_price, call_std_err), (put_price, put_std_err)


# Monte Carlo sample sizes (from 100 to 1,000,000)
M_values = np.logspace(1, np.log10(M), num=10, dtype=int)

# Storage for results
call_means, call_std_errs = [], []
put_means, put_std_errs = [], []

# Run simulations for each M
for M in M_values:
    (call_mean, call_err), (put_mean, put_err) = compute_option_prices(M=M)
    call_means.append(call_mean)
    call_std_errs.append(call_err)
    put_means.append(put_mean)
    put_std_errs.append(put_err)

# Convert to NumPy arrays
call_means = np.array(call_means)
call_std_errs = np.array(call_std_errs)
put_means = np.array(put_means)
put_std_errs = np.array(put_std_errs)

st.subheader("Option Prices and Monte Carlo Convergence")
# Print final estimates
col1, col2 = st.columns(2)
col1.metric("Call Price", value=f"₹{call_means[-1]:.2f}", delta=f"±{1.96*call_std_errs[-1]:.2f}")
col2.metric("Final Put Price", value=f"₹{put_means[-1]:.2f}", delta=f"±{1.96*put_std_errs[-1]:.2f}")

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot call prices with error bars
ax.errorbar(M_values, call_means, yerr=1.96 * call_std_errs, 
            fmt='-o', capsize=4, label='Call Price', color='blue')

# Plot put prices with error bars
ax.errorbar(M_values, put_means, yerr=1.96 * put_std_errs, 
            fmt='-s', capsize=4, label='Put Price', color='red')

# Formatting
ax.set_xscale('log')
ax.set_xlabel("Number of Sample Paths ", fontsize=12)
ax.set_ylabel("Option Price (₹)", fontsize=12)
ax.set_title("Monte Carlo Convergence for Call/Put Prices\n(95% Confidence Intervals)", fontsize=14)
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)

fig.tight_layout()
st.pyplot(fig)
#-------------------------------------------------------------------------------------------------------

# Comparison with Black_Scholes Model

st.subheader("Comparison with Black-Scholes Model")

def bs_option_price(S0=S0, K=K, T=n/252, r=risk_free_rate, sigma=np.exp(h_bar)):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return call_price, put_price

T=n/252
r=risk_free_rate
σ = np.exp(h_bar)
bs_call_price, bs_put_price = bs_option_price(S0, K, T, r, σ)


with st.expander("Black-Scholes Model Details"):
    st.markdown(r"""
    The **Black-Scholes (BS)** model assumes constant volatility, which limits its ability to capture real-world market behavior such as volatility clustering and heavy tails. In contrast, the **LMRSV** model introduces a stochastic and mean-reverting volatility process, making it more robust for realistic option pricing.


    | Feature                       | Black-Scholes Model                     | LMRSV Model                      |
    |------------------------------|------------------------------------------|---------------------------------------------|
    | **Volatility**               | Constant $\sigma$                    | Time-varying: $\sigma_t = \exp(h_t)$   |
    | **Volatility Dynamics**      | None                                     | Mean-reverting stochastic process           |
    | **Asset Returns**            | Lognormal with constant variance         | Lognormal with stochastic variance          |
    | **Market Realism**           | Less realistic in volatile markets       | Captures volatility clustering              |
    | **Greeks & Hedging**         | Analytically tractable                   | Requires simulation                         |
    | **Option Pricing**           | Closed-form (Black-Scholes formula)      | Monte Carlo Simulation                      |

    #### Black-Scholes Formula for Call & Put Options
    """)

    st.latex(r'C = S_0 \, \Phi(d_1) - K e^{-rT} \, \Phi(d_2)')

    st.latex(r'P = K e^{-rT} \, \Phi(-d_2) - S_0 \, \Phi(-d_1)')

    st.latex(r'd_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}')

    st.markdown(r"""
    - $S_0$ : Current asset price  
    - $K$ : Strike price  
    - $r$ : Risk-free rate  
    - $T$ : Time to maturity  
    - $\sigma$ : Constant volatility  
    - $\Phi(\cdot)$ : Standard normal Cumulative Distribution Function

    In contrast, my model replaces $\sigma$  with a **random, evolving volatility** $\sigma_t = \exp(h_t)$ and simulates paths over time to estimate option values numerically.
    """)


# --- Display in Streamlit ---
col1, col2 = st.columns(2)
col1.metric("Call Price", value=f"₹{bs_call_price:.2f}")
col2.metric("Put Price", value=f"₹{bs_put_price:.2f}")

#-----------------------------------------------------------------------------------------------

# Limitations & Possible Improvements
st.subheader("Limitations & Possible Improvements ")
st.warning("Section yet to be added")