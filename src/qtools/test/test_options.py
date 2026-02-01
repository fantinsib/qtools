r"""
  ___        _   _                   _____         _       
 / _ \ _ __ | |_(_) ___  _ __  ___  |_   _|__  ___| |_ ___ 
| | | | '_ \| __| |/ _ \| '_ \/ __|   | |/ _ \/ __| __/ __|
| |_| | |_) | |_| | (_) | | | \__ \   | |  __/\__ \ |_\__ \
 \___/| .__/ \__|_|\___/|_| |_|___/   |_|\___||___/\__|___/
      |_| 
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from qtools.black_scholes import black_scholes
from qtools.options import (
    MarketState,
    OptionBook,
    OptionContract,
    OptionQuote,
    OptionType,
    time_to_mat,
    to_quote,
    make_book
)


def _make_contract(maturity="2030-01-01", T=1.0):
    return OptionContract(
        type=OptionType.EUROPEAN_CALL,
        K=100.0,
        maturity=maturity,
        T=T,
        ticker="TEST",
    )


def _make_market():
    return MarketState(S=100.0, r=0.02)


def test_option_contract_validation():
    with pytest.raises(ValueError):
        OptionContract(type=None, K=100.0, maturity="2030-01-01")
    with pytest.raises(ValueError):
        OptionContract(type=OptionType.EUROPEAN_CALL, K=None, maturity="2030-01-01")
    with pytest.raises(ValueError):
        OptionContract(type=OptionType.EUROPEAN_CALL, K=100.0, maturity=None)


def test_market_state_validation():
    with pytest.raises(ValueError):
        MarketState(S=None, r=0.01)
    with pytest.raises(ValueError):
        MarketState(S=100.0, r=None)


def test_option_quote_properties_and_repr():
    contract = _make_contract()
    market = _make_market()
    quote = OptionQuote(contract=contract, market_state=market, iv=0.2, market_price=10.0)

    assert quote.K == contract.K
    assert quote.S == market.S
    assert quote.T == contract.T
    assert quote.r == market.r
    assert quote.type == contract.type
    assert quote.maturity == contract.maturity
    assert "OptionQuote(" in repr(quote)


def test_option_quote_validation():
    contract = _make_contract()
    market = _make_market()
    with pytest.raises(ValueError):
        OptionQuote(contract=None, market_state=market)
    with pytest.raises(ValueError):
        OptionQuote(contract=contract, market_state=None)


def test_time_to_mat_deterministic():
    maturity = "2030-01-01"
    today = "2029-01-01"
    expiry = datetime.strptime(maturity, "%Y-%m-%d")
    now = datetime.strptime(today, "%Y-%m-%d")
    expected = (expiry - now).total_seconds() / (365.0 * 24.0 * 3600.0)

    assert time_to_mat(maturity, today) == pytest.approx(expected)


def test_time_to_mat_invalid_format_raises():
    with pytest.raises(ValueError):
        time_to_mat("01-01-2030")


def test_to_quote_from_dataframe():
    df = pd.DataFrame(
        {
            "maturity": ["2030-01-01"],
            "type": [OptionType.EUROPEAN_CALL],
            "strike": [105.0],
            "bid": [9.0],
            "ask": [11.0],
            "S": [100.0],
        }
    )
    quotes = to_quote(df, r=0.01, valuation_date="2029-01-01", ticker="TEST")

    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.contract.K == 105.0
    assert quote.market_state.S == 100.0
    assert quote.market_price == pytest.approx(10.0)


def test_option_book_dropna_iv():
    contract = _make_contract()
    market = _make_market()
    quotes = [
        OptionQuote(contract=contract, market_state=market, iv=0.2, market_price=10.0),
        OptionQuote(contract=contract, market_state=market, iv=np.nan, market_price=10.0),
        OptionQuote(contract=contract, market_state=market, iv=np.inf, market_price=10.0),
    ]
    book = OptionBook(quotes)
    cleaned = book.dropna_iv()

    assert len(cleaned.quote_list) == 1
    assert cleaned.quote_list[0].iv == pytest.approx(0.2)


def test_option_book_validation_errors():
    with pytest.raises(ValueError):
        OptionBook(None)
    with pytest.raises(ValueError):
        OptionBook([None])


def test_option_book_compute_iv():
    contract = _make_contract(T=1.0)
    market = _make_market()
    sigma = 0.25
    market_price = black_scholes(contract.type, market.S, contract.K, contract.T, sigma, market.r)
    quote = OptionQuote(contract=contract, market_state=market, market_price=market_price)
    book = OptionBook([quote])

    updated = book.compute_iv()

    assert updated.quote_list[0].iv == pytest.approx(sigma, rel=1e-6)


def test_make_book_no_T():
    
    call = OptionType.EUROPEAN_CALL
    S = [100] * 3
    K = [90, 105, 120]
    maturities = ["2026-01-01","2026-01-01","2026-01-01"]
    r = [0.05]*3
    bid = [10,9,8]
    ask = [11,10,9]
    valuation_date = "2025-01-01"

    book = make_book(call, S, K, maturities, r, bid, ask, valuation_date)

    assert(isinstance(book, OptionBook))
    assert(book.S[1]==100)
    assert(book.K[2]==120)
    mp = [10.5, 9.5, 8.5] 
    assert(book.market_prices[x] == mp[x] for x in range(len(book)))

def test_make_book_with_T():
    
    call = OptionType.EUROPEAN_CALL
    S = [100] * 3
    K = [90, 105, 120]
    maturities = ["2026-01-01","2026-01-01","2026-01-01"]
    r = [0.05]*3
    bid = [10,9,8]
    ask = [11,10,9]
    valuation_date = "2025-01-01"
    T = [0.2,0.4,0.8]

    book = make_book(call, S, K, maturities, r, bid, ask, valuation_date, T)

    assert(isinstance(book, OptionBook))
    assert(book.S[1]==100)
    assert(book.K[2]==120)
    mp = [10.5, 9.5, 8.5] 
    assert(book.market_prices[x] == mp[x] for x in range(len(book)))