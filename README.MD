# qtools

Small Python utilities for quantitative analysis

## Install

```bash
git clone https://github.com/your-org/qtools.git
cd qtools
python -m pip install -e .
```

## Quick start

```python
from qtools.black_scholes import black_scholes
from qtools.options import OptionType

price = black_scholes(
    OptionType.EUROPEAN_CALL,
    S=100.0,
    K=100.0,
    T=1.0,
    sigma=0.2,
    r=0.05,
)
print(price)
```

## Options module overview

The `qtools.options` module provides lightweight data classes to represent
option contracts and market snapshots.

### OptionType

Enum for the option type:

- `OptionType.EUROPEAN_CALL`
- `OptionType.EUROPEAN_PUT`

### OptionContract

Immutable description of a contract.

Fields:
- `type`: `OptionType`
- `K`: strike price
- `maturity`: expiry date as `YYYY-MM-DD`
- `T`: optional time to maturity in years
- `ticker`: optional underlying ticker

### MarketState

Immutable snapshot of the market state:

- `S`: spot price
- `r`: risk-free rate

### OptionQuote

Immutable snapshot of a quote combining a contract and market state.

Fields:
- `contract`: `OptionContract`
- `market_state`: `MarketState`
- `iv`: optional implied volatility
- `market_price`: optional observed price
- `model_price`: optional model price

Convenience properties are available (`K`, `S`, `T`, `r`, `type`, `maturity`).

### OptionBook

A small container for multiple quotes. Supports:

- `compute_iv()` to compute implied vol for each quote
- `dropna_iv()` to remove invalid IV values
- vectorized accessors like `K`, `T`, `iv`, `S`, `r`

### Helpers

The module includes utilities to compute time to maturity:

- `time_to_mat(maturity, today=None)`
- `compute_T(quote, valuation_date=None)`

## Notes

- Currently WIP : features may change
- Python 3.11+
