# Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(layout="wide")

# Binomial Pricing Function
def binomial_op_price(S0, K, T, r, sigma, N, type):
    dt = T / N  # length of each time step
    u = np.exp(sigma * np.sqrt(dt))  # up factor
    d = 1 / u  # down factor
    p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral probability
    disc = np.exp((-r) * dt)  # discount factor per time step

    # Price Tree
    price_tree = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        for j in range(i + 1):
            price_tree[j, i] = S0 * (d ** j) * (u ** (i - j))

    # Option Value
    op_values = np.zeros([N + 1, N + 1])
    if type == "C":
        op_values[:, N] = np.maximum(np.zeros(N + 1), price_tree[:, N] - K)
    elif type == "P":
        op_values[:, N] = np.maximum(np.zeros(N + 1), K - price_tree[:, N])

    # Backward induction to calculate option price
    for i in np.arange(N - 1, -1, -1):
        for j in np.arange(0, i + 1):
            op_values[j, i] = disc * (p * op_values[j, i + 1] + (1 - p) * op_values[j + 1, i + 1])

    return op_values[0, 0]

# Heatmap Function 
def generate_heatmap(S_values, vol_values, K, T, r, N, type):
    option_prices = np.zeros((len(vol_values), len(S_values)))

    for i, sigma in enumerate(vol_values):
        for j, S in enumerate(S_values):
            option_prices[i, j] = np.round(binomial_op_price(S, K, T, r, sigma, N, type=type), 2)

    return option_prices

# Streamlit input elements for parameters
st.title("Binomial Option Pricing Model")

S = st.number_input("Stock Price (S)", min_value=50, max_value=150, value=100)
K = st.number_input("Strike Price (K)", min_value=50, max_value=150, value=99)
T = st.number_input("Time to Expiration (T, in years)", min_value=0.1, max_value=2.0, value=1.0)
r = st.number_input("Risk-Free Rate (r)", min_value=0.01, max_value=0.2, value=0.06)
sigma = st.number_input("Volatility (σ)", min_value=0.1, max_value=0.5, value=0.2)
N = st.slider("Number of Steps (N)", min_value=10, max_value=200, value=50)


call_price = round(binomial_op_price(S, K, T, r, sigma, N, type="C"), 2)
put_price = round(binomial_op_price(S, K, T, r, sigma, N, type="P"), 2)

# displaying call and put prices 
st.metric(label="Call Option Price", value=f"${call_price}", delta=None)
st.metric(label="Put Option Price", value=f"${put_price}", delta=None)

# sliders for stock price range and volatility range
S_min, S_max = st.slider("Stock Price Range", 50, 150, (80, 120))
vol_min, vol_max = st.slider("Volatility Range (σ)", 0.1, 0.5, (0.1, 0.3))

S_values = np.linspace(S_min, S_max, 10)  # Stock price range
vol_values = np.linspace(vol_min, vol_max, 10)  # Volatility range

# Calculate option prices for the heatmaps
call_prices = generate_heatmap(S_values, vol_values, K, T, r, N, type="C")
put_prices = generate_heatmap(S_values, vol_values, K, T, r, N, type="P")

# plot heatmaps
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.heatmap(call_prices, annot=True, xticklabels=np.round(S_values, 2), yticklabels=np.round(vol_values, 2), cmap="RdYlGn", ax=ax[0])
ax[0].set_title('Call Option Prices Heatmap')
ax[0].set_xlabel('Stock Price')
ax[0].set_ylabel('Volatility')

sns.heatmap(put_prices, annot=True, xticklabels=np.round(S_values, 2), yticklabels=np.round(vol_values, 2), cmap="RdYlGn", ax=ax[1])
ax[1].set_title('Put Option Prices Heatmap')
ax[1].set_xlabel('Stock Price')
ax[1].set_ylabel('Volatility')

st.pyplot(fig)