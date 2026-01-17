# qtools

This repository contains a collection of utilities for working with European options. It provides structured representations for option contracts, market states, and quotes, tools to retrieve option chain data via yfinance and transform it into a consistent internal format, implementations of the Black–Scholes model for both scalar and vectorized pricing, a one-dimensional root-finding procedure for implied volatility computation, and functions for modeling and calibrating SVI total variance using least-squares methods.

## Dependencies

The following libraries are required:

- Python 3.10+
- `numpy`
- `scipy`
- `pandas`
- `yfinance`



## Module Overview

### `options.py`

Defines the core data model and utilities for handling option data.

**Main classes**

- `OptionContract`  
  Represents a European option contract.
  - `type`: `OptionType` (call or put)
  - `K`: strike
  - `maturity`: maturity date as string
  - `T`: time to maturity in years (optional)
  - `ticker`: underlying ticker (optional)

- `MarketState`  
  Represents the market state.
  - `S`: spot price
  - `r`: risk-free rate

- `OptionQuote`  
  Represents a single option quote.
  - `contract`: `OptionContract`
  - `market_state`: `MarketState`
  - optional fields: `iv`, `market_price`, `model_price`, `bid`, `ask`

- `OptionBook`  
  Container for a list of `OptionQuote` objects.  
  Provides vectorized accessors and filtering utilities.

**Key functions**

- `fetch_option_data(ticker, maturities, o_type)`  
  Fetches option chains from Yahoo Finance for selected maturities and option type.

- `to_quote(df, r, valuation_date=None, ticker=None)`  
  Converts a fetched DataFrame into a list of `OptionQuote`.

- `compute_T(quote, valuation_date=None)`  
  Computes time to maturity in years and returns an updated `OptionQuote`.

- `time_to_mat(maturity, today=None)`  
  Converts a maturity date into a year fraction.

---

### `black_scholes.py`

Implements Black–Scholes pricing and implied volatility.

**Pricing**

- `black_scholes_price(contract, market, sigma)`  
  Prices a single European option using an `OptionContract` and `MarketState`.

- `black_scholes(o_type, S, K, T, sigma, r)`  
  Scalar Black–Scholes pricing with input validation.

- `price_bs(o_type, S, K, T, sigma, r)`  
  Vectorized Black–Scholes pricing for NumPy arrays (single option type for the batch).

**Implied volatility**

- `iv_solver(market_price, contract, market, sigma_min=1e-6, sigma_max=5, strict=False)`  
  Computes implied volatility using Brent’s method with basic no-arbitrage checks.

- `compute_iv(market_price, quote, sigma_min=1e-6, sigma_max=5)`  
  Returns a new `OptionQuote` with the implied volatility filled in.

---

### `numerical.py`

Numerical helpers.

- `norm_cdf(x)`  
  Standard normal cumulative distribution function implemented with `singledispatch`:
  - scalar input → scalar output
  - NumPy array input → vectorized output

- `finite_difference(f_x, x, degree=1, axis=0)`  
  Finite-difference approximation of first or second derivatives on possibly non-uniform grids.

---

### `svi.py`

SVI (Stochastic Volatility Inspired) total variance model and calibration.

- `svi_total_variance(lm, a, b, rho, m, sigma)`  
  Vectorized SVI total variance function.

- `SVIParam(a, b, rho, m, sigma)`  
  Container for SVI parameters with basic validity checks.

- `SVISlice(a, b, rho, m, sigma, T)`  
  SVI parameters associated with a given maturity.

- `fit_svi(initial, k, w_target, ...)`  
  Least-squares calibration of SVI parameters to market total variance.

- `svi_error(book, fit, metric="mse")`  
  Computes an error metric between market total variance and an SVI fit.

---

### `errors.py`

Defines custom exceptions and warnings used across the library:
- data fetching errors,
- implied volatility solver errors and warnings,
- SVI parameter and calibration errors.

---

## Usage Examples

### Fetch option data and build an `OptionBook`

```python
from dupire_tools.options import fetch_option_data, to_quote, OptionBook, OptionType

df = fetch_option_data(
    ticker="AAPL",
    maturities=["2026-03-20", "2026-06-19"],
    o_type=OptionType.EUROPEAN_CALL
)

quotes = to_quote(df, r=0.02, ticker="AAPL")
book = OptionBook(quotes)
```

### Compute implied volatility for a book

```python
book_iv = book.compute_iv(keep_nan=True)
book_iv_clean = book.compute_iv(keep_nan=False)
```

### Vectorized pricing 

```python
import numpy as np
from dupire_tools.black_scholes import price_bs
from dupire_tools.options import OptionType

S = np.array([100.0, 101.0, 102.0])
K = np.array([100.0, 100.0, 100.0])
T = np.array([0.25, 0.25, 0.25])
sigma = np.array([0.2, 0.21, 0.22])
r = np.array([0.02, 0.02, 0.02])

prices = price_bs(
    OptionType.EUROPEAN_CALL,
    S=S, K=K, T=T, sigma=sigma, r=r
)
```
### Fit SVI parameters on a maturity slice

```python
from dupire_tools.svi import SVIParam, fit_svi

T0 = book_iv.unique_T()[0]
slice_book = book_iv[book_iv.T == T0]

k = slice_book.LM
w = slice_book.W

initial = SVIParam(a=0.01, b=0.1, rho=0.0, m=0.0, sigma=0.2)
fit = fit_svi(initial=initial, k=k, w_target=w)
```