# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:06:32 2020

@author: Pavan
"""


"""


Inputs used by all models
Parameter	Description
option_type	Put/Call Indicator Single character, "c" indicates a call; "p" a put
fs	Price of Underlying FS is generically used, but for specific models, the following abbreviations may be used: F = Forward Price, S = Spot Price)
x	Strike Price
t	Time to Maturity This is in years (1.0 = 1 year, 0.5 = six months, etc)
r	Risk Free Interest Rate Interest rates (0.10 = 10% interest rate
v	Implied Volatility Annualized implied volatility (1=100% annual volatility, 0.34 = 34% annual volatility
Inputs used by some models
Parameter	Description
b	Cost of Carry This is only found in internal implementations, but is identical to the cost of carry (b) term commonly found in academic option pricing literature
q	Continuous Dividend Used in Merton and American models; Internally, this is converted into cost of carry, b, with formula b = r-q
rf	Foreign Interest Rate Only used GK model; this functions similarly to q
t_a	Asian Start Used for Asian options; This is the time that starts the averaging period (TA=0 means that averaging starts immediately). As TA approaches T, the Asian value should become very close to the Black76 Value
cp	Option Price Used in the implied vol calculations; This is the price of the call or put observed in the market


Outputs
All of the option pricing functions return an array. The first element of the array is the value of the option, the other elements are the greeks which measure the sensitivity of the option to changes in inputs. The greeks are used primarily for risk-management purposes.

Output	Description
[0]	Value
[1]	Delta Sensitivity of Value to changes in price
[2]	Gamma Sensitivity of Delta to changes in price
[3]	Theta Sensitivity of Value to changes in time to expiration (annualized). To get a daily Theta, divide by 365
[4]	Vega Sensitivity of Value to changes in Volatility
[5]	Rho Sensitivity of Value to changes in risk-free rates.
The implied volatility functions return a single value (the implied volatility).

Acceptable Range for inputs
All of the inputs are bounded. While many of these functions will work with inputs outside of these bounds, 
they haven't been tested and are generally believed to be uncommon. 
The pricer will return an exception to the caller if an out-of-bounds input is
 used. If that was a valid input, the code below will need to be modified to
 allow wider inputs and the testing section updated to test that the models work
 under the widened inputs.

"""

import math
import numpy as np
from scipy.stats import norm
from scipy.stats import mvn

# Developer can toggle _DEBUG to True for more messages
# normally this is set to False
_DEBUG = False

def _debug(debug_input):
    if (__name__ == "__main__") and (_DEBUG == True):
        print(debug_input)
        
        
# This class defines the Exception that gets thrown when invalid input is placed into the GBS function
class GBS_InputError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)
# This class defines the Exception that gets thrown when invalid input is placed into the GBS function

class GBS_InputError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)
        
# This class contains the limits on inputs for GBS models
# It is not intended to be part of this module's public interface
class _GBS_Limits:
    # An GBS model will return an error if an out-of-bound input is input
    MAX32 = 2147483248.0

    MIN_T = 1.0 / 1000.0  # requires some time left before expiration
    MIN_X = 0.01
    MIN_FS = 0.01

    # Volatility smaller than 0.5% causes American Options calculations
    # to fail (Number to large errors).
    # GBS() should be OK with any positive number. Since vols less
    # than 0.5% are expected to be extremely rare, and most likely bad inputs,
    # _gbs() is assigned this limit too
    MIN_V = 0.005

    MAX_T = 100
    MAX_X = MAX32
    MAX_FS = MAX32

    # Asian Option limits
    # maximum TA is time to expiration for the option
    MIN_TA = 0

    # This model will work with higher values for b, r, and V. However, such values are extremely uncommon. 
    # To catch some common errors, interest rates and volatility is capped to 100%
    # This reason for 1 (100%) is mostly to cause the library to throw an exceptions 
    # if a value like 15% is entered as 15 rather than 0.15)
    MIN_b = -1
    MIN_r = -1

    MAX_b = 1
    MAX_r = 1
    MAX_V = 1

# Inputs: option_type = "p" or "c", fs = price of underlying, x = strike, t = time to expiration, r = risk free rate
#         b = cost of carry, v = implied volatility
# Outputs: value, delta, gamma, theta, vega, rho
def _gbs(option_type, fs, x, t, r, b, v):
    _debug("Debugging Information: _gbs()")
    # -----------
    # Test Inputs (throwing an exception on failure)
    _gbs_test_inputs(option_type, fs, x, t, r, b, v)

    # -----------
    # Create preliminary calculations
    t__sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt

    if option_type == "c":
        # it's a call
        _debug("     Call Option")
        value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
        delta = math.exp((b - r) * t) * norm.cdf(d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = x * t * math.exp(-r * t) * norm.cdf(d2)
    else:
        # it's a put
        _debug("     Put Option")
        value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
        delta = -math.exp((b - r) * t) * norm.cdf(-d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)

    _debug("     d1= {0}\n     d2 = {1}".format(d1, d2))
    _debug("     delta = {0}\n     gamma = {1}\n     theta = {2}\n     vega = {3}\n     rho={4}".format(delta, gamma,
                                                                                                        theta, vega,
                                                                                                        rho))
    
    return value, delta, gamma, theta, vega, rho
    
# ------------------------------
# This function verifies that the Call/Put indicator is correctly entered
def _test_option_type(option_type):
    if (option_type != "c") and (option_type != "p"):
        raise GBS_InputError("Invalid Input option_type ({0}). Acceptable value are: c, p".format(option_type))
        
        
# This function makes sure inputs are OK
# It throws an exception if there is a failure
def _gbs_test_inputs(option_type, fs, x, t, r, b, v):
    # -----------
    # Test inputs are reasonable
    _test_option_type(option_type)

    if (x < _GBS_Limits.MIN_X) or (x > _GBS_Limits.MAX_X):
        raise GBS_InputError(
            "Invalid Input Strike Price (X). Acceptable range for inputs is {1} to {2}".format(x, _GBS_Limits.MIN_X,
                                                                                               _GBS_Limits.MAX_X))

    if (fs < _GBS_Limits.MIN_FS) or (fs > _GBS_Limits.MAX_FS):
        raise GBS_InputError(
            "Invalid Input Forward/Spot Price (FS). Acceptable range for inputs is {1} to {2}".format(fs,
                                                                                                      _GBS_Limits.MIN_FS,
                                                                                                      _GBS_Limits.MAX_FS))

    if (t < _GBS_Limits.MIN_T) or (t > _GBS_Limits.MAX_T):
        raise GBS_InputError(
            "Invalid Input Time (T = {0}). Acceptable range for inputs is {1} to {2}".format(t, _GBS_Limits.MIN_T,
                                                                                             _GBS_Limits.MAX_T))

    if (b < _GBS_Limits.MIN_b) or (b > _GBS_Limits.MAX_b):
        raise GBS_InputError(
            "Invalid Input Cost of Carry (b = {0}). Acceptable range for inputs is {1} to {2}".format(b,
                                                                                                      _GBS_Limits.MIN_b,
                                                                                                      _GBS_Limits.MAX_b))

    if (r < _GBS_Limits.MIN_r) or (r > _GBS_Limits.MAX_r):
        raise GBS_InputError(
            "Invalid Input Risk Free Rate (r = {0}). Acceptable range for inputs is {1} to {2}".format(r,
                                                                                                       _GBS_Limits.MIN_r,
                                                                                                       _GBS_Limits.MAX_r))

    if (v < _GBS_Limits.MIN_V) or (v > _GBS_Limits.MAX_V):
        raise GBS_InputError(
            "Invalid Input Implied Volatility (V = {0}). Acceptable range for inputs is {1} to {2}".format(v,
                                                                                                           _GBS_Limits.MIN_V,
                                                                                                           _GBS_Limits.MAX_V))



    
def _bjerksund_stensland_2002(fs, x, t, r, b, v):
    # -----------
    # initialize output
    # using GBS greeks (TO DO: update greek calculations)
    my_output = _gbs("c", fs, x, t, r, b, v)

    e_value = my_output[0]
    delta = my_output[1]
    gamma = my_output[2]
    theta = my_output[3]
    vega = my_output[4]
    rho = my_output[5]

    # debugging for calculations
    _debug("-----")
    _debug("Debug Information: _Bjerksund_Stensland_2002())")

    # If b >= r, it is never optimal to exercise before maturity
    # so we can return the GBS value
    if b >= r:
        _debug("     Returning GBS value")
        return e_value, delta, gamma, theta, vega, rho

    # -----------
    # Create preliminary calculations
    v2 = v ** 2
    t1 = 0.5 * (math.sqrt(5) - 1) * t
    t2 = t

    beta_inside = ((b / v2 - 0.5) ** 2) + 2 * r / v2
    # forcing the inside of the sqrt to be a positive number
    beta_inside = abs(beta_inside)
    beta = (0.5 - b / v2) + math.sqrt(beta_inside)
    b_infinity = (beta / (beta - 1)) * x
    b_zero = max(x, (r / (r - b)) * x)
    
    h1 = -(b * t1 + 2 * v * math.sqrt(t1)) * ((x ** 2) / ((b_infinity - b_zero) * b_zero))
    h2 = -(b * t2 + 2 * v * math.sqrt(t2)) * ((x ** 2) / ((b_infinity - b_zero) * b_zero))

    i1 = b_zero + (b_infinity - b_zero) * (1 - math.exp(h1))
    i2 = b_zero + (b_infinity - b_zero) * (1 - math.exp(h2))

    alpha1 = (i1 - x) * (i1 ** (-beta))
    alpha2 = (i2 - x) * (i2 ** (-beta))
    
        # check for immediate exercise
    if fs >= i2:
        value = fs - x
    else:
        # Perform the main calculation    
        value = (alpha2 * (fs ** beta)
                 - alpha2 * _phi(fs, t1, beta, i2, i2, r, b, v)
                 + _phi(fs, t1, 1, i2, i2, r, b, v)
                 - _phi(fs, t1, 1, i1, i2, r, b, v)
                 - x * _phi(fs, t1, 0, i2, i2, r, b, v)
                 + x * _phi(fs, t1, 0, i1, i2, r, b, v)
                 + alpha1 * _phi(fs, t1, beta, i1, i2, r, b, v)
                 - alpha1 * _psi(fs, t2, beta, i1, i2, i1, t1, r, b, v)
                 + _psi(fs, t2, 1, i1, i2, i1, t1, r, b, v)
                 - _psi(fs, t2, 1, x, i2, i1, t1, r, b, v)
                 - x * _psi(fs, t2, 0, i1, i2, i1, t1, r, b, v)
                 + x * _psi(fs, t2, 0, x, i2, i1, t1, r, b, v))

    # in boundary conditions, this approximation can break down
    # Make sure option value is greater than or equal to European value
    value = max(value, e_value)

    # -----------
    # Return Data
    return value, delta, gamma, theta, vega, rho



# ---------------------------
# American Option Intermediate Calculations

# -----------
# The Psi() function used by _Bjerksund_Stensland_2002 model
def _psi(fs, t2, gamma, h, i2, i1, t1, r, b, v):
    vsqrt_t1 = v * math.sqrt(t1)
    vsqrt_t2 = v * math.sqrt(t2)

    bgamma_t1 = (b + (gamma - 0.5) * (v ** 2)) * t1
    bgamma_t2 = (b + (gamma - 0.5) * (v ** 2)) * t2

    d1 = (math.log(fs / i1) + bgamma_t1) / vsqrt_t1
    d3 = (math.log(fs / i1) - bgamma_t1) / vsqrt_t1

    d2 = (math.log((i2 ** 2) / (fs * i1)) + bgamma_t1) / vsqrt_t1
    d4 = (math.log((i2 ** 2) / (fs * i1)) - bgamma_t1) / vsqrt_t1

    e1 = (math.log(fs / h) + bgamma_t2) / vsqrt_t2
    e2 = (math.log((i2 ** 2) / (fs * h)) + bgamma_t2) / vsqrt_t2
    e3 = (math.log((i1 ** 2) / (fs * h)) + bgamma_t2) / vsqrt_t2
    e4 = (math.log((fs * (i1 ** 2)) / (h * (i2 ** 2))) + bgamma_t2) / vsqrt_t2

    tau = math.sqrt(t1 / t2)
    lambda1 = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * (v ** 2))
    kappa = (2 * b) / (v ** 2) + (2 * gamma - 1)

    psi = math.exp(lambda1 * t2) * (fs ** gamma) * (_cbnd(-d1, -e1, tau)
                                                    - ((i2 / fs) ** kappa) * _cbnd(-d2, -e2, tau)
                                                    - ((i1 / fs) ** kappa) * _cbnd(-d3, -e3, -tau)
                                                    + ((i1 / i2) ** kappa) * _cbnd(-d4, -e4, -tau))
    return psi


# The Phi() function used by _Bjerksund_Stensland_2002 model and the _Bjerksund_Stensland_1993 model
def _phi(fs, t, gamma, h, i, r, b, v):
    d1 = -(math.log(fs / h) + (b + (gamma - 0.5) * (v ** 2)) * t) / (v * math.sqrt(t))
    d2 = d1 - 2 * math.log(i / fs) / (v * math.sqrt(t))

    lambda1 = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * (v ** 2))
    kappa = (2 * b) / (v ** 2) + (2 * gamma - 1)

    phi = math.exp(lambda1 * t) * (fs ** gamma) * (norm.cdf(d1) - ((i / fs) ** kappa) * norm.cdf(d2))


    return phi


# Cumulative Bivariate Normal Distribution
# Primarily called by Psi() function, part of the _Bjerksund_Stensland_2002 model
def _cbnd(a, b, rho):
    # This distribution uses the Genz multi-variate normal distribution 
    # code found as part of the standard SciPy distribution
    lower = np.array([0, 0])
    upper = np.array([a, b])
    infin = np.array([0, 0])
    correl = rho
    error, value, inform = mvn.mvndst(lower, upper, infin, correl)
    return value


# This function tests that two floating point numbers are the same
# Numbers less than 1 million are considered the same if they are within .000001 of each other
# Numbers larger than 1 million are considered the same if they are within .0001% of each other
# User can override the default precision if necessary
def assert_close(value_a, value_b, precision=.000001):
    my_precision = precision

    if (value_a < 1000000.0) and (value_b < 1000000.0):
        my_diff = abs(value_a - value_b)
        my_diff_type = "Difference"
    else:
        my_diff = abs((value_a - value_b) / value_a)
        my_diff_type = "Percent Difference"

    _debug("Comparing {0} and {1}. Difference is {2}, Difference Type is {3}".format(value_a, value_b, my_diff,
                                                                                     my_diff_type))

    if my_diff < my_precision:
        my_result = True
    else:
        my_result = False

    if (__name__ == "__main__") and (my_result == False):
        print("  FAILED TEST. Comparing {0} and {1}. Difference is {2}, Difference Type is {3}".format(value_a, value_b,
                                                                                                       my_diff,
                                                                                                       my_diff_type))
    else:
        print(".")

    return my_result


# -----------
# Generalized American Option Pricer
# This is a wrapper to check inputs and route to the current "best" American option model
def american_option(option_type, fs, x, t, r, v,b=0):
    # -----------
    # Test Inputs (throwing an exception on failure)
    _debug("Debugging Information: _american_option()")
    _gbs_test_inputs(option_type, fs, x, t, r, b, v)

    # -----------
    if option_type == "c":
        # Call Option
        _debug("     Call Option")
        return _bjerksund_stensland_2002(fs, x, t, r, b, v)
    else:
        # Put Option
        _debug("     Put Option")

        # Using the put-call transformation: P(X, FS, T, r, b, V) = C(FS, X, T, -b, r-b, V)
        # WARNING - When reconciling this code back to the B&S paper, the order of variables is different

        put__x = fs
        put_fs = x
        put_b = -b
        put_r = r - b

        # pass updated values into the Call Valuation formula
        return _bjerksund_stensland_2002(put_fs, put__x, t, put_r, put_b, v)




## ---------------------------
## Testing American Options
#if __name__ == "__main__":
#    print("=====================================")
#    print("American Options Testing")
#    print("=====================================")
#
#    print("testing _Bjerksund_Stensland_2002()")
#    # _american_option(option_type, X, FS, T, b, r, V)
#    assert_close(_bjerksund_stensland_2002(fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15)[0], 0.8099, precision=.001)
#    assert_close(_bjerksund_stensland_2002(fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25)[0], 6.7661, precision=.001)
#    assert_close(_bjerksund_stensland_2002(fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35)[0], 15.5137, precision=.001)
#
#    assert_close(_bjerksund_stensland_2002(fs=100, x=90, t=0.5, r=.1, b=0, v=0.15)[0], 10.5400, precision=.001)
#    assert_close(_bjerksund_stensland_2002(fs=100, x=100, t=0.5, r=.1, b=0, v=0.25)[0], 6.7661, precision=.001)
#    assert_close(_bjerksund_stensland_2002(fs=100, x=110, t=0.5, r=.1, b=0, v=0.35)[0], 5.8374, precision=.001)