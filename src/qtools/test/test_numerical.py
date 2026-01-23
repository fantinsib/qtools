r"""
 _            _                                     _           _ 
| |_ ___  ___| |_   _ __  _   _ _ __ ___   ___ _ __(_) ___ __ _| |
| __/ _ \/ __| __| | '_ \| | | | '_ ` _ \ / _ \ '__| |/ __/ _` | |
| ||  __/\__ \ |_  | | | | |_| | | | | | |  __/ |  | | (_| (_| | |
 \__\___||___/\__| |_| |_|\__,_|_| |_| |_|\___|_|  |_|\___\__,_|_|
                                                             
"""

import pytest
import numpy as np

from qtools.numerical import norm_cdf, correlated_brownians


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, rel=0, abs=1e-12)
    assert norm_cdf(1.0) == pytest.approx(0.841344746, rel=1e-9)
    assert norm_cdf(-1.0) == pytest.approx(0.158655254, rel=1e-9)


def test_norm_cdf_symmetry():
    x = 0.7
    assert norm_cdf(x) == pytest.approx(1.0 - norm_cdf(-x), rel=1e-12)

def test_norm_cdf_monotonicity():
    assert norm_cdf(-2.0) < norm_cdf(-1.0) < norm_cdf(0.0) < norm_cdf(1.0) < norm_cdf(2.0)

def test_corr_brownians():
    p = 0.6
    W = correlated_brownians(p, 1.0, 50000)  

    dW1 = np.diff(np.concatenate(([0.0], W[0])))
    dW2 = np.diff(np.concatenate(([0.0], W[1])))

    corr = np.corrcoef(dW1, dW2)[0, 1]
    assert corr == pytest.approx(p, rel=0.02)