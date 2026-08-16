import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from pricing import (
    black_scholes_greeks,
    black_scholes_price,
    calculate_greeks,
    price_option,
)


sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Binomial Option Pricing",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0.8rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def calculate_heatmaps(
    stock_min,
    stock_max,
    volatility_min,
    volatility_max,
    strike_price,
    time_to_expiry,
    risk_free_rate,
    steps,
    exercise_style,
):
    """Calculate the values used in the two heatmaps."""

    stock_values = np.linspace(stock_min, stock_max, 9)
    volatility_values = np.linspace(volatility_min, volatility_max, 7)
    call_prices = np.empty((len(volatility_values), len(stock_values)))
    put_prices = np.empty_like(call_prices)

    for row, volatility in enumerate(volatility_values):
        for column, stock_price in enumerate(stock_values):
            common_inputs = {
                "stock_price": stock_price,
                "strike_price": strike_price,
                "time_to_expiry": time_to_expiry,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "steps": steps,
                "exercise_style": exercise_style,
            }
            call_prices[row, column] = price_option(
                **common_inputs,
                option_type="call",
            )
            put_prices[row, column] = price_option(
                **common_inputs,
                option_type="put",
            )

    return stock_values, volatility_values, call_prices, put_prices


def create_heatmap_figure(
    stock_values,
    volatility_values,
    call_prices,
    put_prices,
):
    """Create the call and put sensitivity heatmaps."""

    colour_min = min(call_prices.min(), put_prices.min())
    colour_max = max(call_prices.max(), put_prices.max())
    stock_labels = [f"{value:.0f}" for value in stock_values]
    volatility_labels = [
        f"{100 * value:.0f}" for value in volatility_values[::-1]
    ]

    heatmap_options = {
        "annot": True,
        "annot_kws": {"fontsize": 8},
        "cmap": "RdYlGn",
        "fmt": ".2f",
        "linewidths": 0.5,
        "linecolor": "white",
        "vmin": colour_min,
        "vmax": colour_max,
        "xticklabels": stock_labels,
        "yticklabels": volatility_labels,
    }

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    figure.patch.set_facecolor("white")

    sns.heatmap(
        call_prices[::-1],
        ax=axes[0],
        cbar=False,
        **heatmap_options,
    )
    sns.heatmap(
        put_prices[::-1],
        ax=axes[1],
        cbar_kws={"label": "Option value"},
        **heatmap_options,
    )

    axes[0].set_title("Call option values")
    axes[1].set_title("Put option values")

    for axis in axes:
        axis.set_xlabel("Stock price")
    axes[0].set_ylabel("Volatility (%)")
    axes[1].set_ylabel("")

    figure.tight_layout()
    return figure


@st.cache_data(show_spinner=False)
def calculate_convergence(
    stock_price,
    strike_price,
    time_to_expiry,
    risk_free_rate,
    volatility,
):
    """Calculate CRR prices as the number of periods increases."""

    period_values = np.array([5, 10, 25, 50, 100, 200, 500, 1000])
    valid_periods = []
    call_values = []
    put_values = []

    for periods in period_values:
        common_inputs = {
            "stock_price": stock_price,
            "strike_price": strike_price,
            "time_to_expiry": time_to_expiry,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
            "steps": int(periods),
        }
        try:
            call_value = price_option(**common_inputs, option_type="call")
            put_value = price_option(**common_inputs, option_type="put")
        except ValueError:
            continue

        valid_periods.append(periods)
        call_values.append(call_value)
        put_values.append(put_value)

    return (
        np.array(valid_periods),
        np.array(call_values),
        np.array(put_values),
        len(period_values),
    )


def create_convergence_figure(
    period_values,
    call_values,
    put_values,
    black_scholes_call,
    black_scholes_put,
):
    """Create the call and put convergence plots."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    figure.patch.set_facecolor("white")

    axes[0].plot(period_values, call_values, marker="o", label="CRR price")
    axes[0].axhline(
        black_scholes_call,
        color="black",
        linestyle="--",
        label="Black-Scholes",
    )
    axes[0].set_title("European call convergence")

    axes[1].plot(
        period_values,
        put_values,
        marker="o",
        color="C1",
        label="CRR price",
    )
    axes[1].axhline(
        black_scholes_put,
        color="black",
        linestyle="--",
        label="Black-Scholes",
    )
    axes[1].set_title("European put convergence")

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(period_values)
        axis.set_xticklabels(period_values)
        axis.set_xlabel("Number of periods (log scale)")
        axis.set_ylabel("Option value")
        axis.grid(alpha=0.2)
        axis.legend()

    figure.tight_layout()
    return figure


st.title("Binomial Option Pricing Model")
st.caption(
    "Price European and American call and put options using the "
    "Cox-Ross-Rubinstein binomial model."
)


# model inputs
with st.container(border=True):
    st.subheader("Model inputs")

    first_row = st.columns(4)
    stock_price = first_row[0].number_input(
        "Stock price",
        min_value=1.0,
        value=100.0,
        step=1.0,
        format="%.2f",
    )
    strike_price = first_row[1].number_input(
        "Strike price",
        min_value=1.0,
        value=99.0,
        step=1.0,
        format="%.2f",
    )
    time_to_expiry = first_row[2].number_input(
        "Time to expiry (years)",
        min_value=0.01,
        value=1.0,
        step=0.05,
        format="%.2f",
    )
    exercise_style_input = first_row[3].selectbox(
        "Exercise style",
        ("European", "American"),
    )

    second_row = st.columns(3)
    risk_free_rate_input = second_row[0].number_input(
        "Risk-free rate (%)",
        min_value=-5.0,
        max_value=25.0,
        value=6.0,
        step=0.25,
        format="%.2f",
    )
    volatility_input = second_row[1].number_input(
        "Volatility (%)",
        min_value=1.0,
        max_value=150.0,
        value=20.0,
        step=1.0,
        format="%.2f",
    )
    steps = second_row[2].slider(
        "Number of periods",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

risk_free_rate = risk_free_rate_input / 100
volatility = volatility_input / 100
exercise_style = exercise_style_input.lower()

common_inputs = {
    "stock_price": stock_price,
    "strike_price": strike_price,
    "time_to_expiry": time_to_expiry,
    "risk_free_rate": risk_free_rate,
    "volatility": volatility,
    "steps": steps,
    "exercise_style": exercise_style,
}

try:
    call_price = price_option(**common_inputs, option_type="call")
    put_price = price_option(**common_inputs, option_type="put")
except ValueError as error:
    st.error(str(error))
    st.stop()


# option values
st.subheader("Option values")
call_column, put_column = st.columns(2)

if exercise_style == "european":
    black_scholes_call = black_scholes_price(
        stock_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility,
        option_type="call",
    )
    black_scholes_put = black_scholes_price(
        stock_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility,
        option_type="put",
    )
    call_benchmark = (
        f"Black-Scholes benchmark: {black_scholes_call:,.2f}\n\n"
        f"Absolute difference: {abs(call_price - black_scholes_call):,.4f}"
    )
    put_benchmark = (
        f"Black-Scholes benchmark: {black_scholes_put:,.2f}\n\n"
        f"Absolute difference: {abs(put_price - black_scholes_put):,.4f}"
    )
else:
    european_call = price_option(
        **{**common_inputs, "exercise_style": "european"},
        option_type="call",
    )
    european_put = price_option(
        **{**common_inputs, "exercise_style": "european"},
        option_type="put",
    )
    call_benchmark = (
        f"European CRR benchmark: {european_call:,.2f}\n\n"
        f"Early-exercise premium: {call_price - european_call:,.4f}"
    )
    put_benchmark = (
        f"European CRR benchmark: {european_put:,.2f}\n\n"
        f"Early-exercise premium: {put_price - european_put:,.4f}"
    )

with call_column:
    with st.container(border=True):
        st.metric("Call option", f"${call_price:,.2f}")
        st.caption(call_benchmark)

with put_column:
    with st.container(border=True):
        st.metric("Put option", f"${put_price:,.2f}")
        st.caption(put_benchmark)


# analysis tabs
st.subheader("Analysis")
if exercise_style == "european":
    sensitivity_tab, greeks_tab, convergence_tab = st.tabs(
        ("Sensitivity", "Greeks", "Convergence")
    )
else:
    sensitivity_tab, greeks_tab = st.tabs(("Sensitivity", "Greeks"))

with sensitivity_tab:
    st.write(
        "See how the option values change as the stock price and volatility change."
    )

    range_columns = st.columns(2)
    stock_slider_max = max(300, int(stock_price * 2))
    stock_range = range_columns[0].slider(
        "Stock price range",
        min_value=1,
        max_value=stock_slider_max,
        value=(
            max(1, int(stock_price * 0.8)),
            max(2, int(stock_price * 1.2)),
        ),
    )
    volatility_range_input = range_columns[1].slider(
        "Volatility range (%)",
        min_value=1,
        max_value=150,
        value=(10, 40),
    )

    try:
        (
            stock_values,
            volatility_values,
            call_prices,
            put_prices,
        ) = calculate_heatmaps(
            stock_range[0],
            stock_range[1],
            volatility_range_input[0] / 100,
            volatility_range_input[1] / 100,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            steps,
            exercise_style,
        )

        heatmap_figure = create_heatmap_figure(
            stock_values,
            volatility_values,
            call_prices,
            put_prices,
        )
        st.pyplot(heatmap_figure, use_container_width=True)
        plt.close(heatmap_figure)
    except ValueError as error:
        st.warning(
            "The sensitivity range includes inputs that do not produce a valid "
            f"tree: {error}"
        )

with greeks_tab:
    st.write(
        "Vega and Rho are shown for a one percentage-point change. "
        "Theta is shown per calendar day."
    )

    try:
        call_greeks = calculate_greeks(**common_inputs, option_type="call")
        put_greeks = calculate_greeks(**common_inputs, option_type="put")

        greek_names = {
            "delta": "Delta",
            "gamma": "Gamma",
            "vega": "Vega",
            "theta": "Theta",
            "rho": "Rho",
        }

        if exercise_style == "european":
            black_scholes_call_greeks = black_scholes_greeks(
                stock_price,
                strike_price,
                time_to_expiry,
                risk_free_rate,
                volatility,
                option_type="call",
            )
            black_scholes_put_greeks = black_scholes_greeks(
                stock_price,
                strike_price,
                time_to_expiry,
                risk_free_rate,
                volatility,
                option_type="put",
            )
            greek_rows = [
                {
                    "Greek": name,
                    "CRR call": f"{call_greeks[key]:.4f}",
                    "BS call": f"{black_scholes_call_greeks[key]:.4f}",
                    "CRR put": f"{put_greeks[key]:.4f}",
                    "BS put": f"{black_scholes_put_greeks[key]:.4f}",
                }
                for key, name in greek_names.items()
            ]
        else:
            greek_rows = [
                {
                    "Greek": name,
                    "American call": f"{call_greeks[key]:.4f}",
                    "American put": f"{put_greeks[key]:.4f}",
                }
                for key, name in greek_names.items()
            ]

        st.dataframe(
            greek_rows,
            hide_index=True,
            use_container_width=True,
        )
    except ValueError as error:
        st.warning(f"The Greeks could not be estimated for these inputs: {error}")

if exercise_style == "european":
    with convergence_tab:
        st.write(
            "Compare the CRR prices with their Black-Scholes limits as the tree "
            "becomes finer."
        )
        (
            convergence_periods,
            call_convergence,
            put_convergence,
            total_period_counts,
        ) = calculate_convergence(
            stock_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            volatility,
        )

        if len(convergence_periods) > 0:
            convergence_figure = create_convergence_figure(
                convergence_periods,
                call_convergence,
                put_convergence,
                black_scholes_call,
                black_scholes_put,
            )
            st.pyplot(convergence_figure, use_container_width=True)
            plt.close(convergence_figure)

            if len(convergence_periods) < total_period_counts:
                st.caption(
                    "Some coarse trees are omitted because their risk-neutral "
                    "probability is not valid for these inputs."
                )
        else:
            st.warning("No valid convergence values could be calculated.")


with st.expander("Model assumptions"):
    st.markdown(
        """
        - The stock follows the Cox-Ross-Rubinstein up/down process.
        - Volatility and the risk-free rate stay constant.
        - Markets are frictionless, with no transaction costs or taxes.
        - European options can only be exercised at expiry.
        - American options are checked for early exercise at every node.
        """
    )
