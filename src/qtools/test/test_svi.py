r"""
 ______     _____   _            _       
/ ___\ \   / /_ _| | |_ ___  ___| |_ ___ 
\___ \\ \ / / | |  | __/ _ \/ __| __/ __|
 ___) |\ V /  | |  | ||  __/\__ \ |_\__ \
|____/  \_/  |___|  \__\___||___/\__|___/
"""

import pytest 
import numpy as np
from qtools import svi
from qtools.errors import SVIFitError, SVIFitWarning, SVIParamError
from qtools.options import MarketState, OptionBook, OptionContract, OptionQuote, OptionType


def test_svi_param_validation():
    with pytest.raises(SVIParamError):
        svi.SVIParam(a=1, b=0, rho=-0.5, m=0, sigma=0.2)
    with pytest.raises(SVIParamError):
        svi.SVIParam(a=1, b=1, rho=-1.5, m=0, sigma=0.2)
    with pytest.raises(SVIParamError):
        svi.SVIParam(a=1, b=1, rho=0.5, m=0, sigma=-0.2)


def test_scalar_vector_match():
    scalar = svi.svi_w_scalar(lm=0.2, a=0.1, b=0.5, rho=0.2, m=0.0, sigma=0.3)
    vec = svi.svi_total_variance(lm=np.array([0.2]), a=0.1, b=0.5, rho=0.2, m=0.0, sigma=0.3)
    assert scalar == vec[0]


def test_svi_slice_validation():
    with pytest.raises(SVIParamError):
        svi.SVISlice(a=0.1, b=0.0, rho=0.2, m=0.0, sigma=0.3, T=1.0)
    with pytest.raises(SVIParamError):
        svi.SVISlice(a=0.1, b=0.2, rho=1.0, m=0.0, sigma=0.3, T=1.0)
    with pytest.raises(SVIParamError):
        svi.SVISlice(a=0.1, b=0.2, rho=0.2, m=0.0, sigma=0.0, T=1.0)
    with pytest.raises(SVIParamError):
        svi.SVISlice(a=0.1, b=0.2, rho=0.2, m=0.0, sigma=0.3, T=0.0)


def test_svi_param_to_slice():
    params = svi.SVIParam(a=0.1, b=0.2, rho=-0.3, m=0.0, sigma=0.4)
    slice_ = params.to_slice(T=0.5)
    assert slice_.T == 0.5
    assert slice_.params == params


def test_fit_svi_recovers_params():
    true_params = svi.SVIParam(a=0.02, b=0.4, rho=-0.2, m=0.1, sigma=0.3)
    k = np.linspace(-0.3, 0.3, 7)
    w_target = svi.svi_total_variance(k, *true_params)
    fit = svi.fit_svi(initial=true_params, k=k, w_target=w_target)
    assert fit.a == pytest.approx(true_params.a, rel=1e-6)
    assert fit.b == pytest.approx(true_params.b, rel=1e-6)
    assert fit.rho == pytest.approx(true_params.rho, rel=1e-6)
    assert fit.m == pytest.approx(true_params.m, rel=1e-6)
    assert fit.sigma == pytest.approx(true_params.sigma, rel=1e-6)


def test_fit_svi_weight_shape_error():
    params = svi.SVIParam(a=0.02, b=0.4, rho=-0.2, m=0.1, sigma=0.3)
    k = np.linspace(-0.3, 0.3, 5)
    w_target = svi.svi_total_variance(k, *params)
    weight_i = np.ones(len(k) + 1)
    with pytest.raises(SVIFitError):
        svi.fit_svi(initial=params, k=k, w_target=w_target, weight_i=weight_i)


def test_fit_svi_lam_warns_without_norm():
    params = svi.SVIParam(a=0.02, b=0.4, rho=-0.2, m=0.1, sigma=0.3)
    k = np.linspace(-0.3, 0.3, 5)
    w_target = svi.svi_total_variance(k, *params)
    with pytest.warns(SVIFitWarning):
        svi.fit_svi(initial=params, k=k, w_target=w_target, lam=0.1, norm_residuals=False)


def test_svi_error_zero_for_perfect_fit():
    params = svi.SVIParam(a=0.01, b=0.2, rho=-0.1, m=0.0, sigma=0.25)
    market = MarketState(S=100.0, r=0.0)
    maturity = "2030-01-01"
    T = 1.0
    strikes = [90.0, 100.0, 110.0]
    quotes = []
    for K in strikes:
        contract = OptionContract(type=OptionType.EUROPEAN_CALL, K=K, maturity=maturity, T=T)
        lm = np.log(K / (market.S * np.exp(market.r * T)))
        w = svi.svi_w_scalar(lm, *params)
        iv = np.sqrt(w / T)
        quotes.append(OptionQuote(contract=contract, market_state=market, iv=iv, market_price=10.0))
    book = OptionBook(quotes)
    assert svi.svi_error(book, params, metric="mse") == pytest.approx(0.0, abs=1e-12)
