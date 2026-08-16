"""Option pricing functions used by the notebook and Streamlit app."""

from math import erf, exp, isfinite, log, sqrt

import numpy as np


VALID_OPTION_TYPES = {"call", "put"}
VALID_EXERCISE_STYLES = {"european", "american"}


def _check_inputs(
    stock_price,
    strike_price,
    time_to_expiry,
    risk_free_rate,
    volatility,
    steps,
    option_type,
    exercise_style,
):
    """Check the inputs needed by the CRR model."""

    # check that the numerical inputs are usable
    numerical_inputs = {
        "Stock price": stock_price,
        "Strike price": strike_price,
        "Time to expiry": time_to_expiry,
        "Risk-free rate": risk_free_rate,
        "Volatility": volatility,
    }
    valid_number_types = (int, float, np.integer, np.floating)

    for input_name, input_value in numerical_inputs.items():
        if (
            isinstance(input_value, (bool, np.bool_))
            or not isinstance(input_value, valid_number_types)
            or not isfinite(input_value)
        ):
            raise ValueError(f"{input_name} must be a finite number.")

    if stock_price <= 0:
        raise ValueError("Stock price must be greater than zero.")
    if strike_price <= 0:
        raise ValueError("Strike price must be greater than zero.")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be greater than zero.")
    if volatility <= 0:
        raise ValueError("Volatility must be greater than zero.")
    if (
        not isinstance(steps, (int, np.integer))
        or isinstance(steps, (bool, np.bool_))
        or steps < 1
    ):
        raise ValueError("Number of steps must be a positive integer.")
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError("Option type must be 'call' or 'put'.")
    if exercise_style not in VALID_EXERCISE_STYLES:
        raise ValueError("Exercise style must be 'european' or 'american'.")


def _normal_cdf(value):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _normal_pdf(value):
    """Standard normal probability density function."""
    return exp(-0.5 * value**2) / sqrt(2.0 * np.pi)


def _price_and_levels(
    stock_price,
    strike_price,
    time_to_expiry,
    risk_free_rate,
    volatility,
    steps,
    option_type="call",
    exercise_style="european",
):
    """Price an option and keep the first two tree levels for the Greeks."""

    # use one format for inputs such as "Call" and "call"
    if not isinstance(option_type, str) or not isinstance(exercise_style, str):
        raise ValueError("Option type and exercise style must be text values.")
    option_type = option_type.lower()
    exercise_style = exercise_style.lower()
    _check_inputs(
        stock_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility,
        steps,
        option_type,
        exercise_style,
    )

    # set up one period of the CRR tree
    time_step = time_to_expiry / steps
    up_factor = exp(volatility * sqrt(time_step))
    down_factor = 1.0 / up_factor
    growth_factor = exp(risk_free_rate * time_step)
    probability_up = (growth_factor - down_factor) / (up_factor - down_factor)

    # the no-arbitrage condition needs a probability strictly between 0 and 1
    if not 0.0 < probability_up < 1.0:
        raise ValueError(
            "The risk-neutral probability must be strictly between 0 and 1. "
            "Check the inputs or increase the number of steps."
        )

    # calculate the stock prices at expiry
    discount_factor = exp(-risk_free_rate * time_step)
    up_moves = np.arange(steps + 1)
    stock_values = stock_price * up_factor**up_moves * down_factor ** (steps - up_moves)

    # calculate the option payoffs at expiry
    if option_type == "call":
        option_values = np.maximum(stock_values - strike_price, 0.0)
    else:
        option_values = np.maximum(strike_price - stock_values, 0.0)

    # save the levels later used for Delta, Gamma and Theta
    saved_levels = {}
    if steps <= 2:
        saved_levels[steps] = option_values.copy()

    # work backwards from expiry to the first node
    for step in range(steps - 1, -1, -1):
        option_values = discount_factor * (
            probability_up * option_values[1:]
            + (1.0 - probability_up) * option_values[:-1]
        )

        # compare holding the option with exercising it now
        if exercise_style == "american":
            up_moves = np.arange(step + 1)
            stock_values = (
                stock_price * up_factor**up_moves * down_factor ** (step - up_moves)
            )
            if option_type == "call":
                exercise_values = np.maximum(stock_values - strike_price, 0.0)
            else:
                exercise_values = np.maximum(strike_price - stock_values, 0.0)
            option_values = np.maximum(option_values, exercise_values)

        if step <= 2:
            saved_levels[step] = option_values.copy()

    return (
        float(option_values[0]),
        saved_levels,
        up_factor,
        down_factor,
        time_step,
    )


def price_option(
    stock_price,
    strike_price,
    time_to_expiry,
    risk_free_rate,
    volatility,
    steps,
    option_type="call",
    exercise_style="european",
):
    """Price an option with the Cox-Ross-Rubinstein binomial model."""
    price, _, _, _, _ = _price_and_levels(
        stock_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility,
        steps,
        option_type,
        exercise_style,
    )
    return price


def calculate_greeks(
    stock_price,
    strike_price,
    time_to_expiry,
    risk_free_rate,
    volatility,
    steps,
    option_type="call",
    exercise_style="european",
):
    """Calculate the main Greeks from the binomial tree."""

    if (
        not isinstance(steps, (int, np.integer))
        or isinstance(steps, (bool, np.bool_))
        or steps < 2
    ):
        raise ValueError("At least two steps are needed to calculate the Greeks.")

    # keep the shared values together when the option is repriced
    common_inputs = {
        "stock_price": stock_price,
        "strike_price": strike_price,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "steps": steps,
        "option_type": option_type,
        "exercise_style": exercise_style,
    }
    base_price, levels, up_factor, down_factor, time_step = _price_and_levels(
        **common_inputs
    )

    # Delta uses the two option values after the first period
    stock_down = stock_price * down_factor
    stock_up = stock_price * up_factor
    delta = (levels[1][1] - levels[1][0]) / (stock_up - stock_down)

    # Gamma uses the change in Delta across the second period
    stock_down_down = stock_price * down_factor**2
    stock_up_down = stock_price
    stock_up_up = stock_price * up_factor**2
    delta_down = (levels[2][1] - levels[2][0]) / (stock_up_down - stock_down_down)
    delta_up = (levels[2][2] - levels[2][1]) / (stock_up_up - stock_up_down)
    gamma = (delta_up - delta_down) / ((stock_up_up - stock_down_down) / 2.0)

    # reprice after a small volatility change to estimate Vega
    volatility_bump = min(0.01, volatility / 2.0)
    volatility_up = price_option(
        **{**common_inputs, "volatility": volatility + volatility_bump}
    )
    volatility_down = price_option(
        **{**common_inputs, "volatility": volatility - volatility_bump}
    )
    vega = (volatility_up - volatility_down) / (2.0 * volatility_bump) * 0.01

    # use the middle node after two periods to estimate daily Theta
    theta = (levels[2][1] - base_price) / (2.0 * time_step * 365.0)

    # reprice after a small rate change to estimate Rho
    rate_bump = 0.001
    higher_rate = price_option(
        **{**common_inputs, "risk_free_rate": risk_free_rate + rate_bump}
    )
    lower_rate = price_option(
        **{**common_inputs, "risk_free_rate": risk_free_rate - rate_bump}
    )
    rho = (higher_rate - lower_rate) / (2.0 * rate_bump) * 0.01

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def black_scholes_price(
    stock_price,
    strike_price,
    time_to_expiry,
    risk_free_rate,
    volatility,
    option_type="call",
):
    """Price a European option using the Black-Scholes formula."""

    if not isinstance(option_type, str):
        raise ValueError("Option type must be a text value.")
    option_type = option_type.lower()
    _check_inputs(
        stock_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility,
        1,
        option_type,
        "european",
    )

    # calculate the two values used in the Black-Scholes formula
    volatility_over_life = volatility * sqrt(time_to_expiry)
    d1 = (
        log(stock_price / strike_price)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / volatility_over_life
    d2 = d1 - volatility_over_life

    # discount the strike back to today
    discounted_strike = strike_price * exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        return stock_price * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)

    return discounted_strike * _normal_cdf(-d2) - stock_price * _normal_cdf(-d1)


def black_scholes_greeks(
    stock_price,
    strike_price,
    time_to_expiry,
    risk_free_rate,
    volatility,
    option_type="call",
):
    """Calculate Black-Scholes Greeks for a European option."""

    if not isinstance(option_type, str):
        raise ValueError("Option type must be a text value.")
    option_type = option_type.lower()
    _check_inputs(
        stock_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility,
        1,
        option_type,
        "european",
    )

    # reuse the Black-Scholes values needed by each Greek
    volatility_over_life = volatility * sqrt(time_to_expiry)
    d1 = (
        log(stock_price / strike_price)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / volatility_over_life
    d2 = d1 - volatility_over_life
    discounted_strike = strike_price * exp(-risk_free_rate * time_to_expiry)
    density_d1 = _normal_pdf(d1)

    # Gamma and Vega use the same formula for calls and puts
    gamma = density_d1 / (stock_price * volatility_over_life)
    vega = stock_price * density_d1 * sqrt(time_to_expiry) * 0.01

    # Delta, Theta and Rho depend on the option type
    if option_type == "call":
        delta = _normal_cdf(d1)
        annual_theta = (
            -stock_price * density_d1 * volatility / (2.0 * sqrt(time_to_expiry))
            - risk_free_rate * discounted_strike * _normal_cdf(d2)
        )
        rho = discounted_strike * time_to_expiry * _normal_cdf(d2) * 0.01
    else:
        delta = -_normal_cdf(-d1)
        annual_theta = (
            -stock_price * density_d1 * volatility / (2.0 * sqrt(time_to_expiry))
            + risk_free_rate * discounted_strike * _normal_cdf(-d2)
        )
        rho = -discounted_strike * time_to_expiry * _normal_cdf(-d2) * 0.01

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(annual_theta / 365.0),
        "rho": float(rho),
    }
