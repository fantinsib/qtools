r"""
 ______     _____   _____           _     
/ ___\ \   / /_ _| |_   _|__   ___ | |___ 
\___ \\ \ / / | |    | |/ _ \ / _ \| / __|
 ___) |\ V /  | |    | | (_) | (_) | \__ \
|____/  \_/  |___|   |_|\___/ \___/|_|___/
                                          

SVI model Notes :

log_moneyness: log moneyness (log(K/F))
a: minimum total variance
b: slope. must be stricly positive
rho: skew. must be in (-1,1)
m: center, location of smile minimum
sigma: curvature; must be strictly positive

w(k) = a + b(rho(k-m) + sqrt[(k-m)^2 + sigma^2]

"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import least_squares

def compute_svi(a, b, rho, m, sigma) -> float:
    """
    Compute total implied volatility from the 
    SVI parameters

    Parameters
    ----------
        a: float
            Minimum total variance
        b: float
            Slope. must be stricly positive
        rho: float 
            Skew. must be in (-1,1)
        m: float
            Center, location of smile minimum
        sigma: float
            Curvature; must be strictly positive

    """


@dataclass(frozen=True)
class SVISlice:
    """
    Set of SVI parameters that correspond to a certain maturity

    Attributes:
        log_moneyness: log moneyness (log(K/F))
        a: minimum total variance
        b: slope. must be stricly positive
        rho: skew. must be in (-1,1)
        m: center, location of smile minimum
        sigma: curvature; must be strictly positive
    """
    a: float
    b : float
    rho: float 
    m: float 
    sigma: float 
    T: float


def fit_svi():

    def residuals()