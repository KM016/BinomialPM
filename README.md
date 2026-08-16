# Binomial Option Pricing Model

A Python implementation of the Cox-Ross-Rubinstein binomial model for pricing and analysing European and American options.

> Personal Project<br>
> Date: 09/2024

## Project overview

This project implements the Cox-Ross-Rubinstein (CRR) model for European and American call and put options. The model builds a recombining stock-price tree and works backwards from the payoff at expiry to calculate the option value today.

The project also uses the Black-Scholes model as an analytical benchmark for European options. This makes it possible to examine how the binomial price converges as the number of periods increases and to compare estimates of Delta, Gamma, Vega, Theta and Rho.

An interactive Streamlit app brings the pricing and analysis together. Its inputs can be changed to compare option values, inspect the early-exercise premium and explore sensitivity to the stock price and volatility.

## Objectives

The project was built to:

- implement the CRR model without relying on an option-pricing library;
- price European and American call and put options;
- compare European binomial prices with Black-Scholes values;
- calculate the five main option Greeks;
- examine convergence as the tree becomes finer;
- visualise sensitivity to stock price and volatility; and
- test the main mathematical relationships and input checks.

## Cox-Ross-Rubinstein model

The time to expiry is divided into $N$ periods of length

$$
\Delta t=\frac{T}{N}.
$$

At each period, the stock price moves up by a factor $u$ or down by a factor $d$:

$$
u=e^{\sigma\sqrt{\Delta t}}, \qquad d=\frac{1}{u}.
$$

The risk-neutral probability of an upward movement is

$$
p=\frac{e^{r\Delta t}-d}{u-d}.
$$

Starting from the call or put payoff at expiry, the value at each earlier node is calculated by backward induction:

$$
V=e^{-r\Delta t}\left(pV_{up}+(1-p)V_{down}\right).
$$

For an American option, the continuation value is compared with the value from exercising immediately. The larger of the two is retained at each node.

## Black-Scholes benchmark

The Black-Scholes model provides a closed-form price for the European options under the same stock price, strike, expiry, interest-rate and volatility assumptions. It is used here as a benchmark rather than as a replacement for the tree.

For the default inputs, $S_0=100$, $K=99$, $T=1$, $r=6\%$, $\sigma=20\%$ and 50 periods, the results are:

| Option | CRR price | Black-Scholes price | Absolute difference |
| --- | ---: | ---: | ---: |
| Call | 11.5464 | 11.5443 | 0.0022 |
| Put | 4.7811 | 4.7790 | 0.0022 |

The convergence plots repeat the CRR calculation over increasingly fine trees and compare the results with these analytical limits.

## Option Greeks

The project calculates:

- **Delta:** sensitivity to a change in the stock price;
- **Gamma:** sensitivity of Delta to a further change in the stock price;
- **Vega:** sensitivity to a one percentage-point change in volatility;
- **Theta:** change in value per calendar day; and
- **Rho:** sensitivity to a one percentage-point change in the risk-free rate.

Delta, Gamma and Theta are estimated from the first two levels of the binomial tree. Vega and Rho are estimated by repricing after small changes in volatility and the interest rate. For European options, the app displays the corresponding analytical Black-Scholes Greeks alongside the CRR estimates.

## Streamlit app

The app includes:

- inputs for the stock price, strike, expiry, risk-free rate, volatility and number of periods;
- a choice between European and American exercise;
- call and put values displayed together;
- Black-Scholes comparisons for European options;
- European CRR comparisons and early-exercise premiums for American options;
- a table of the five main Greeks;
- stock-price and volatility sensitivity heatmaps; and
- CRR convergence plots for European calls and puts.

[Open the Streamlit app](https://binomialpm.streamlit.app/)

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Validation

The automated tests cover:

- the original numerical pricing example;
- put-call parity;
- convergence to Black-Scholes;
- agreement between CRR and Black-Scholes Greeks;
- the early-exercise value of an American put;
- equality of American and European call values in the no-dividend model;
- expected price movements as the stock price and volatility change; and
- invalid, non-positive and non-finite inputs.

The current test suite contains 13 tests.

## Tools used

- NumPy for the binomial calculations and backward induction;
- Matplotlib and Seaborn for the sensitivity and convergence plots;
- Streamlit for the interactive interface; and
- Python's `unittest` framework for validation.

## Repository contents

```text
.
├── app.py                 # Streamlit interface and visualisations
├── main.ipynb             # Model explanation and numerical analysis
├── pricing.py             # CRR and Black-Scholes pricing functions
├── requirements.txt       # Python dependencies
└── tests/
    └── test_pricing.py    # Numerical and input-validation tests
```

## Scope and limitations

- The stock is assumed to follow the CRR up/down process.
- Volatility and the continuously compounded risk-free rate remain constant.
- The model does not include dividends, transaction costs, taxes or liquidity effects.
- Black-Scholes is only used as a benchmark for European options.
- Binomial prices can oscillate around the limiting value as the number of periods changes.
- The numerical Greeks depend on the tree resolution and the small changes used for Vega and Rho.
- The project is an educational pricing model rather than a production trading or risk system.
