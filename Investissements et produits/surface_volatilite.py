import pandas as pd
import datetime
import scipy
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import pandas_datareader as web
import numpy as np

from scipy.stats import norm
from scipy import stats
from statistics import mean
from datetime import date
from datetime import timedelta, date
from yahoofinancials import YahooFinancials
from random import randint
from monte_carlo import *
from pandas_datareader.data import Options
from dateutil.parser import parse
from datetime import datetime
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
from functools import partial
from scipy import optimize
import numpy as np
from scipy.interpolate import griddata

class surface_volatilite():

    N_prime = norm.pdf
    N = norm.cdf

    def black_scholes_call(self, S, K, T, r, vol):
        '''
        :param S: Asset price
        :param K: Strike price
        :param T: Time to maturity
        :param r: risk-free rate (treasury bills)
        :param sigma: volatility
        :return: call price
        '''

        d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

    def bs_vega(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    def find_vol(self, target_value, S, K, T, r):
        MAX_ITERATIONS = 200
        PRECISION = 1.0e-5
        sigma = 0.5
        for i in range(0, MAX_ITERATIONS):
            price = self.black_scholes_call(S, K, T, r, sigma)
            vega = self.bs_vega(S, K, T, r, sigma)
            diff = target_value - price  # our root
            if (abs(diff) < PRECISION):
                return sigma
            sigma = sigma + diff/vega # f(x) / f'(x)
        return sigma

if __name__ == '__main__':

    s_v = surface_volatilite()
    m_t = monte_carlo()

    # On prend trois option call européenne
    first_call = s_v.black_scholes_call(m_t.s0,150,1,m_t.r,m_t.volatility)
    second_call = s_v.black_scholes_call(m_t.s0,120,6,m_t.r,m_t.volatility)
    third_call = s_v.black_scholes_call(m_t.s0,220,5,m_t.r,m_t.volatility)

    # Calcul de la volatilité implicite pour chaque option
    #V_market = bs_call(S, K, T, r, vol)
    #implied_vol = find_vol(V_market, S, K, T, r)

    implied_vol_1 = s_v.find_vol(first_call,m_t.s0,150,1,m_t.r)
    implied_vol_2 = s_v.find_vol(second_call,m_t.s0,120,6,m_t.r)
    implied_vol_3 = s_v.find_vol(third_call,m_t.s0,200,5,m_t.r)

    print ('Implied vol for first call : %.2f%%' % (implied_vol_1 * 100))
    print ('Implied vol for second call : %.2f%%' % (implied_vol_2 * 100))
    print ('Implied vol for third call : %.2f%%' % (implied_vol_3 * 100))


