import streamlit as st

st.title("Portfolio")


st.markdown("""
Welcome to my project hub! This platform showcases a collection of my projects spanning **Quantitative Finance**, **Artificial Intelligence / Machine Learning**, and **Data Science**. Each project includes a detailed explanation, code, and interactive components for deeper understanding and experimentation.
""")


# Project 1: Monte Carlo Simulation
st.subheader("Monte Carlo Simulation of Lognormal Mean-Reverting Stochastic Volatility Model for Option Pricing", divider='red')
st.markdown("""
    This project simulates option pricing using a **lognormal mean-reverting stochastic volatility model**, comparing classical Monte Carlo simulation with analytical benchmarks such as the **Black-Scholes** model. It includes real-time **NSE data fetching**, volatility modeling, and pricing visualizations.
    """)


# Project 2: RL Stock Trading
st.subheader("Reinforcement Learning Model (A2C) Based Stock Trading AI Agent", divider='red')
st.markdown("""
    A stock trading strategy trained using the **Advantage Actor-Critic (A2C)** reinforcement learning algorithm on a selected time period of historical data. It includes feature engineering with **technical indicators**, environment design, model training, and live evaluation.
    """)

st.divider()

st.markdown("""
More projects are coming soon, including:
- AI/ML models for financial forecasting
- Quantum algorithms for derivative pricing
- Portfolio optimization tools

Stay tuned!
""")
