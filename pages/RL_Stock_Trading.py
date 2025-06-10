import streamlit as st

# Gym stuff
import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Quant Finance
import yfinance as yf
from finta import TA
#-----------------------------------------------

st.title("Reinforcement Learning Model (A2C)")
st.subheader("Stock Trading AI Agent", divider='red')

nse_symbols = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BEL.NS", "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "ETERNAL.NS", "GRASIM.NS", "HCLTECH.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS"
]



stock = st.sidebar.selectbox(
    "Select NSE stock",
    nse_symbols,
)

# Mapping for display
period_display_map = {
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "10 years": "10y",
    "Max available": "max"
}

# Create selectbox with display labels
selected_display = st.sidebar.selectbox("Select time period for training", list(period_display_map.keys()), index=2)

# Get the actual period value
selected_period = period_display_map[selected_display]


data = yf.Ticker(stock).history(period=selected_period)
data.drop(columns=['Dividends','Stock Splits'], inplace=True)

# finta expects columns to be named: open, high, low, close, volume
data = data.rename(columns={
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

data['return'] = np.log(data['close'] / data['close'].shift(1))

#Create columns for technical indicators & add them to the dataframe
data['RSI'] = TA.RSI(data,14)
data['SMA'] = TA.SMA(data, 20)
data['SMA_L'] = TA.SMA(data, 50)
data['OBV'] = TA.OBV(data)
data['VWAP'] = TA.VWAP(data)
data['EMA'] = TA.EMA(data)
data['ATR'] = TA.ATR(data)
data.fillna(0, inplace=True)
data['momentum'] = data['return'].rolling(5).mean().shift(1)
data['volatility'] = data['return'].rolling(20).std().shift(1)
data['distance'] = (data['close'] - data['close'].rolling(50).mean()).shift(1)

st.subheader("Latest Week's Data")

with st.expander("Technical Indicators Explained"):
    st.markdown(r"""
    Here are the technical indicators calculated for the stock price data:

    - **RSI (Relative Strength Index)**: Measures the speed and change of price movements. It helps identify overbought or oversold conditions by comparing recent gains to losses over a 14-period window.

    - **SMA (Simple Moving Average)**: The average closing price over a given period, smoothing out price fluctuations. Here, a short-term 20-period SMA and a longer-term 50-period SMA are calculated to show trend direction.

    - **OBV (On-Balance Volume)**: A cumulative total of volume that adds volume on up days and subtracts volume on down days. It helps to confirm price trends with volume flow.

    - **VWAP (Volume Weighted Average Price)**: The average price weighted by volume traded, often used as a benchmark to evaluate trade prices during the trading day.

    - **EMA (Exponential Moving Average)**: A moving average that gives more weight to recent prices, making it more responsive to new information.

    - **ATR (Average True Range)**: Measures market volatility by decomposing the entire range of an asset price for that period.

    Additional computed features:

    - **Return**: The logarithmic return between consecutive closing prices, measuring relative price changes.

    - **Momentum**: The average return over the past 5 periods (shifted by 1 to avoid lookahead bias), indicating the strength of recent price movements.

    - **Volatility**: The standard deviation of returns over the past 20 periods (shifted by 1), capturing the variability of price changes.

    - **Distance**: The difference between the current closing price and its 50-period moving average, showing how far price is from its longer-term trend.
    """)

st.dataframe(data.tail())
#--------------------------------------------------------------------------------------


#Perform a simple linear regression direction prediction
st.subheader("Linear Regression to predict Closing price")
with st.expander("Purpose of the predicted_close Column"):
    st.markdown(r"""
    **Trend Information:**
    - The linear regression uses the past 5 lagged close prices to predict the current close price. This captures short-term trends in the data.
    - The predicted value (`predicted_close`) represents the expected price based on recent historical behavior, which can help the RL model anticipate whether the price is likely to rise or fall.

    **Feature Enrichment:**
    - The RL model (A2C in this case) relies on the state representation (observations) to make decisions. By including `predicted_close` as part of the state (in `signal_features`), the model gets an additional signal about the market's directionality beyond raw prices or technical indicators like RSI or SMA.

    **Bias Toward Mean Reversion or Momentum:**
    - If the predicted price is higher than the current price, it might suggest an upward trend (momentum). Conversely, if it's lower, it might indicate a downward trend or mean reversion. The RL model can use this to inform its trading strategy (e.g., buy if the predicted price is higher).
    """)

lags = 5

cols = []
for lag in range(1, lags + 1):
  col = f'lag_{lag}'
  data[col] = data['close'].shift(lag)
  cols.append(col)

data.dropna(inplace=True)

reg = np.linalg.lstsq(data[cols], data['close'], rcond=None)[0]
data['predicted_close'] = np.dot(data[cols], reg)
st.dataframe(data[['close','predicted_close']].tail())
#---------------------------------------------------------------------------------


class MyCustomEnv(StocksEnv):
    # Create a function to properly format data frame to be passed through environment
    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:,'close'].to_numpy()[start:end]
        signal_features = self.df.loc[:, ['open','high','low','volume','return','momentum','volatility','distance','RSI','OBV','SMA','SMA_L','VWAP','EMA','ATR', 'predicted_close']].to_numpy()[start:end]
        return prices, signal_features

    
    #--------------------------------------------------------------------------


def runRLCode():
    #Initialize an environment setting the window size and train data
    window_size = 63
    start_index = window_size
    end_train_index = round(len(data)*0.80)
    end_val_index = len(data)

    temp_env = MyCustomEnv(df=data, window_size=window_size, frame_bound=(start_index, end_train_index))
    env_maker = lambda: temp_env
    env = DummyVecEnv([env_maker])

    #initialize our model and train
    policy_kwargs = dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
    actor_critic = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    actor_critic.learn(total_timesteps=10000)
    #----------------------------------------------------------------------------------------


    #Create a new environment with validation data
    env = MyCustomEnv(df=data, window_size=window_size, frame_bound=(end_train_index, end_val_index))
    obs, _ = env.reset()

    step = 0
    while True:
        obs = obs[np.newaxis, ...]
        action, _states = actor_critic.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated
        if done:
            print("info", info)
            break

    
    #Plot the results
    fig = plt.figure(figsize=(16,9))
    env.render_all()
    #plt.show()
    st.pyplot(fig)


#---------------------------------------------------------------------------------------------------


st.subheader("Reinforcement Learning Model")

with st.expander("Advantage Actor-Critic Model (A2C) Details"):
    st.image("https://www.researchgate.net/profile/Lorenzo-Federici/publication/343639364/figure/fig5/AS:925509202505735@1597669997770/Schematic-of-the-Advantage-Actor-Critic-RL-process-for-a-deterministic-MDP.png", caption="Diagrammatic representation of A2C Model")

    st.markdown(r'''
    #### 1. Core Components of A2C
    **1.1 Actor (Policy Network)** \
    Role: Decides the best action (here - buy, sell, hold) given the current state (stock market data). \
    Output: A probability distribution over actions (e.g., 30% sell, 60% hold, 10% buy). \
    Training: Adjusts its policy to maximize expected rewards (profits in trading). 

    **1.2 Critic (Value Network)** \
    Role: Estimates the expected future reward (value) of being in a given state. \
    Output: A single value (V(s)) representing how good the current state is. \
    Training: Learns to reduce prediction error (difference between estimated and actual rewards). 

    **1.3 Advantage Function** \
    Definition:
    ''')

    st.latex(r'A(s,a)=Q(s,a)−V(s)')

    st.markdown(r'''
    Q(s, a) = Expected reward if we take action a in state s. \
    V(s) = Expected reward from state s (estimated by the Critic). 

    Interpretation: \
    If A(s, a) > 0, action a is better than average. \
    If A(s, a) < 0, action a is worse than average. \
    This reduces variance in training by comparing actions to the average (instead of raw rewards).

    #### 2. Working of A2C Works
    **2.1 Agent Observes State (s)** \
    In my implementation, this includes:
    - Price history (window_size=63).
    - Technical indicators (RSI, SMA, predicted_close, etc.).

    **2.2 Actor Chooses Action (a)** \
    Samples an action (buy/sell/hold) from its policy.

    **2.3 Environment Executes Action & Returns Reward (r)** \
    Reward could be:
    - Positive (Profit from a trade)
    - Negative (Loss from a trade)

    **2.4 Critic Evaluates State (V(s))** \
    Predicts how good the current state is for future rewards.

    **2.5 Advantage is Calculated** \
    Uses N-step returns (rewards from multiple steps) to compute:
    ''')

    st.latex(r'A(s,a) = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)')

    st.markdown(r'''
    Where γ (gamma) is the discount factor (e.g., 0.99).

    **2.6 Update Actor & Critic** \
    Actor Update: Adjusts policy to favor actions with positive advantage. \
    Critic Update: Improves its value predictions to match actual returns.
    ''')

if st.button("Run RL Code"):
    with st.spinner("Running... please wait"):
        runRLCode()


#------------------------------------------------------------------------

# Limitations and Potential Improvements
st.subheader("Limitations and Potential Improvements")
st.warning("Section yet to be added")
