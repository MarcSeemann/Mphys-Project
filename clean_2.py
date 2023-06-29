import uproot
import matplotlib.pyplot as plt
import numpy as np
import boost_histogram as bh
from numba import jit
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import time as ti
import pandas as pd
import scipy.stats as st
from scipy.optimize import least_squares



y_arr = np.array([[1, 2, 3], [4,5,6], [7,8,9]])
y_arr = np.sum(y_arr, axis=0)
print(y_arr)