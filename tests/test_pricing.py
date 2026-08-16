import unittest
from math import exp

from pricing import (
    black_scholes_greeks,
    black_scholes_price,
    calculate_greeks,
    price_option,
)


class TestOptionPricing(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "stock_price": 100,
            "strike_price": 99,
            "time_to_expiry": 1,
            "risk_free_rate": 0.06,
            "volatility": 0.2,
        }

    def test_original_example(self):
        call_price = price_option(**self.inputs, steps=50, option_type="call")
        put_price = price_option(**self.inputs, steps=50, option_type="put")

        self.assertAlmostEqual(call_price, 11.5464348508, places=8)
        self.assertAlmostEqual(put_price, 4.7811236756, places=8)

    def test_black_scholes_example(self):
        call_price = black_scholes_price(**self.inputs, option_type="call")
        put_price = black_scholes_price(**self.inputs, option_type="put")

        self.assertAlmostEqual(call_price, 11.5442802271, places=8)
        self.assertAlmostEqual(put_price, 4.7789690519, places=8)

    def test_put_call_parity(self):
        call_price = price_option(**self.inputs, steps=200, option_type="call")
        put_price = price_option(**self.inputs, steps=200, option_type="put")
        discounted_strike = 99 * exp(-0.06)

        self.assertAlmostEqual(
            call_price - put_price, 100 - discounted_strike, places=10
        )

    def test_binomial_converges_to_black_scholes(self):
        for option_type in ("call", "put"):
            with self.subTest(option_type=option_type):
                binomial_price = price_option(
                    **self.inputs, steps=500, option_type=option_type
                )
                benchmark = black_scholes_price(
                    **self.inputs, option_type=option_type
                )

                self.assertLess(abs(binomial_price - benchmark), 0.01)

    def test_american_put_has_early_exercise_value(self):
        american_inputs = {
            "stock_price": 80,
            "strike_price": 100,
            "time_to_expiry": 1,
            "risk_free_rate": 0.05,
            "volatility": 0.25,
        }
        european_put = price_option(
            **american_inputs,
            steps=500,
            option_type="put",
            exercise_style="european",
        )
        american_put = price_option(
            **american_inputs,
            steps=500,
            option_type="put",
            exercise_style="american",
        )

        self.assertGreater(american_put, european_put)

    def test_american_call_matches_european_call(self):
        european_call = price_option(
            **self.inputs, steps=200, option_type="call", exercise_style="european"
        )
        american_call = price_option(
            **self.inputs, steps=200, option_type="call", exercise_style="american"
        )

        self.assertAlmostEqual(american_call, european_call, places=10)

    def test_prices_move_with_stock_price(self):
        lower_call = price_option(
            **{**self.inputs, "stock_price": 90}, steps=100, option_type="call"
        )
        higher_call = price_option(
            **{**self.inputs, "stock_price": 110}, steps=100, option_type="call"
        )
        lower_put = price_option(
            **{**self.inputs, "stock_price": 90}, steps=100, option_type="put"
        )
        higher_put = price_option(
            **{**self.inputs, "stock_price": 110}, steps=100, option_type="put"
        )

        self.assertGreater(higher_call, lower_call)
        self.assertLess(higher_put, lower_put)

    def test_prices_increase_with_volatility(self):
        for option_type in ("call", "put"):
            with self.subTest(option_type=option_type):
                lower_price = price_option(
                    **{**self.inputs, "volatility": 0.15},
                    steps=200,
                    option_type=option_type,
                )
                higher_price = price_option(
                    **{**self.inputs, "volatility": 0.30},
                    steps=200,
                    option_type=option_type,
                )

                self.assertGreater(higher_price, lower_price)

    def test_binomial_greeks_are_close_to_black_scholes(self):
        tolerances = {
            "delta": 0.002,
            "gamma": 0.0002,
            "vega": 0.002,
            "theta": 0.0002,
            "rho": 0.002,
        }

        for option_type in ("call", "put"):
            binomial = calculate_greeks(
                **self.inputs, steps=500, option_type=option_type
            )
            benchmark = black_scholes_greeks(**self.inputs, option_type=option_type)
            for greek, tolerance in tolerances.items():
                self.assertLess(abs(binomial[greek] - benchmark[greek]), tolerance)

    def test_greek_signs(self):
        call_greeks = calculate_greeks(**self.inputs, steps=200, option_type="call")
        put_greeks = calculate_greeks(**self.inputs, steps=200, option_type="put")

        self.assertGreater(call_greeks["delta"], 0)
        self.assertLess(put_greeks["delta"], 0)
        self.assertGreater(call_greeks["gamma"], 0)
        self.assertGreater(put_greeks["gamma"], 0)
        self.assertGreater(call_greeks["vega"], 0)
        self.assertGreater(put_greeks["vega"], 0)
        self.assertGreater(call_greeks["rho"], 0)
        self.assertLess(put_greeks["rho"], 0)

    def test_non_positive_inputs_raise_an_error(self):
        for input_name in (
            "stock_price",
            "strike_price",
            "time_to_expiry",
            "volatility",
        ):
            with self.subTest(input_name=input_name):
                invalid_inputs = {**self.inputs, input_name: 0}
                with self.assertRaises(ValueError):
                    price_option(
                        **invalid_inputs,
                        steps=50,
                        option_type="call",
                    )

    def test_non_finite_inputs_raise_an_error(self):
        for input_name in (
            "stock_price",
            "strike_price",
            "time_to_expiry",
            "risk_free_rate",
            "volatility",
        ):
            for invalid_value in (float("nan"), float("inf")):
                with self.subTest(
                    input_name=input_name,
                    invalid_value=invalid_value,
                ):
                    invalid_inputs = {**self.inputs, input_name: invalid_value}
                    with self.assertRaises(ValueError):
                        price_option(
                            **invalid_inputs,
                            steps=50,
                            option_type="call",
                        )

    def test_invalid_model_settings_raise_an_error(self):
        with self.assertRaises(ValueError):
            price_option(**self.inputs, steps=50, option_type="other")

        with self.assertRaises(ValueError):
            price_option(
                **self.inputs,
                steps=50,
                option_type="call",
                exercise_style="other",
            )

        for invalid_steps in (0, 1.5, True):
            with self.subTest(steps=invalid_steps):
                with self.assertRaises(ValueError):
                    price_option(
                        **self.inputs,
                        steps=invalid_steps,
                        option_type="call",
                    )

        with self.assertRaises(ValueError):
            price_option(
                stock_price=100,
                strike_price=100,
                time_to_expiry=1,
                risk_free_rate=0.2,
                volatility=0.01,
                steps=1,
                option_type="call",
            )

        for boundary_rate in (-0.2, 0.2):
            with self.subTest(risk_free_rate=boundary_rate):
                with self.assertRaises(ValueError):
                    price_option(
                        stock_price=100,
                        strike_price=100,
                        time_to_expiry=1,
                        risk_free_rate=boundary_rate,
                        volatility=0.2,
                        steps=1,
                        option_type="call",
                    )

        with self.assertRaises(ValueError):
            calculate_greeks(**self.inputs, steps=1, option_type="call")

        with self.assertRaises(ValueError):
            calculate_greeks(**self.inputs, steps="two", option_type="call")


if __name__ == "__main__":
    unittest.main()
