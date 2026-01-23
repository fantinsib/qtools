r"""
 _ __  _   _ _ __ ___   ___ _ __(_) ___ __ _| |
| '_ \| | | | '_ ` _ \ / _ \ '__| |/ __/ _` | |
| | | | |_| | | | | | |  __/ |  | | (_| (_| | |
|_| |_|\__,_|_| |_| |_|\___|_|  |_|\___\__,_|_|                                    
"""
import math
import numpy as np
from typing import Optional, overload
from functools import singledispatch
from scipy.special import erf

@singledispatch
def norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the cumulative distribution function (CDF) of the standard
    normal distribution.

    - if ``x`` is a float, the CDF is evaluated at a single point and a
      scalar is returned;
    - if ``x`` is a NumPy array, the CDF is evaluated element-wise and a
      NumPy array of the same shape is returned.
      
    Parameters
    ----------
    x : float or numpy.ndarray
        Point(s) at which to evaluate the standard normal CDF.

    Returns
    -------
    float or numpy.ndarray
        Value(s) of the standard normal CDF evaluated at ``x``.

    Raises
    ------
    TypeError
        If ``x`` is not a float or a NumPy array.
    """
    raise TypeError("Expected argument must be a float or a np.array")

@norm_cdf.register
def _(x: float)->float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@norm_cdf.register
def _(x:np.ndarray)->np.ndarray:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def finite_difference(f_x: np.ndarray, x: np.ndarray, degree: int = 1, axis:int = 0) -> np.ndarray:
    """
    Derivative of f_x with respect to x

    Parameters
    ----------
    f_x : np.ndarray
        Array of shape (len(x), n) where the first axis matches x.
    x : np.ndarray
        1D grid (monotone increasing) of length f_x.shape[0].
    degree : int
        1 for first derivative, 2 for second derivative.
    axis: int
        The axis along which to compute the derivative.

    Returns
    -------
    np.ndarray
        Derivative array with the same shape as f_x.
    """

    if f_x.shape[axis] != len(x):
        raise ValueError("Array dimension does not match x along the chosen axis.")    
    f = np.moveaxis(f_x, axis, 0)
    d_f = np.empty_like(f)

    if degree == 1:
        d_f[1:-1, ...] = (f[2:, ...] - f[:-2, ...]) / (x[2:, None] - x[:-2, None])
        d_f[0, ...] = (f[1, ...] - f[0, ...]) / (x[1] - x[0])
        d_f[-1, ...] = (f[-1, ...] - f[-2, ...]) / (x[-1] - x[-2])

    elif degree == 2:
        dx_fwd = x[2:] - x[1:-1]
        dx_bwd = x[1:-1] - x[:-2]

        d_f[1:-1, ...] = 2.0 * (
            (f[2:, ...] - f[1:-1, ...]) / dx_fwd[:, None]
            - (f[1:-1, ...] - f[:-2, ...]) / dx_bwd[:, None]
        ) / (dx_fwd + dx_bwd)[:, None]

        d_f[0, ...] = d_f[1, ...]
        d_f[-1, ...] = d_f[-2, ...]

    else:
        raise ValueError("degree must be 1 or 2")

    return np.moveaxis(d_f, 0, axis)

def correlated_brownians(p: float, T: float, n: int) ->np.ndarray:
    """
    Generates two correlated brownian motion path

    Parameters
    ----------
    p : float
        The correlation of brownien increments comprised in [-1;1]
    T : float
        Time interval
    n : int
        Number of discretization steps over T

    Returns
    -------
    np.ndarray
        Array of shape (2,n) containing the brownian paths ; W0 = 0 not included
    """
    if (p < -1) or (p>1): raise ValueError("correlated_brownians : correlation must be in [-1,1]")
    if (T<=0): raise ValueError("correlated_brownians : T must be positive")
    if (n<=0) or not isinstance(n, int): raise ValueError("correlated_brownians : n must be a positive int")

    dt = T/n
    dW1 = np.random.normal(0, np.sqrt(dt), n)
    dW2 = p*dW1 + np.sqrt(1-p**2)*np.random.normal(0,np.sqrt(dt),n)

    return np.array([dW1.cumsum(), dW2.cumsum()])

