import pandas as pd
import datetime
import scipy
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import pandas_datareader as web
import numpy as np

from scipy import stats
from statistics import mean
from datetime import date
from datetime import timedelta, date
from yahoofinancials import YahooFinancials
from random import randint
from marche_aleatoire import *
from surface_volatilite import *


class monte_carlo():
    
    symbol = "AAPL"

    # Create the object representing the symbol of interest
    equity = yf.Ticker(symbol)

    current_price = equity.info['currentPrice']
    print(f"Current price of {symbol} - ${current_price}")

    cdf = stats.norm(0, 1).cdf

    # Optional seed
    np.random.seed(12345678)

    # Parameters
    s0 = current_price	          	# Actual price
    drift = 0.0016273		      # Drift term (daily)
    volatility = 0.088864	  	# Volatility (daily)
    t_ = 365 		            	# Total periods in a year
    r = 0.033 			          # Risk free rate (yearly)
    days = 2			            # Days until option expiration
    N = 100000		          	# Number of Monte Carlo trials
    zero_trials = 0		      	# Number of trials where the option payoff = 0
    k = 100				            # Strike price

    avg = 0			            	# Temporary variable to be assigned to the sum
                                    # of the simulated payoffs
    T = 1                           # maturity T

    # Stocastic walk
    # This function calculates the stochastic integral after periods
    # and returns the final price.
    def stoc_walk(self, p, dr, vol, periods):
        w = np.random.normal(0,1,size=periods)
        for i in range(periods):
            p += dr*p + w[i]*vol*p
        return p
    
    def long_call(self, last_value_walk, k):
    # Long Put Payoff = max(Strike Price - Stock Price, 0)     # If we are long a call, we would only elect to call if the current stock price is greater than     # the strike price on our option
        payoff_list = []     
        for i in range(len(last_value_walk)):
            P = max(last_value_walk[i] - k, 0)
            payoff_list.append(P)
        return payoff_list
    
    # Modèle de Black Scholes
    def d1(self, S, K, T, r, q, o):
        return (np.log(S / K) + ((r-q) + 0.5 * o ** 2) * T) / (o * np.sqrt(T))
    
    def d2(self, S, K, T, r,q, o):
        return self.d1(S, K, T, r,q, o) - o * np.sqrt(T)

    def call(self, S, K, T, r,q, o):
        return S *np.exp(-q*T)* self.cdf(self.d1(S, K, T, r,q, o)) - K * np.exp(-r * T) * self.cdf(self.d2(S, K, T, r,q, o))
   
if __name__ == '__main__':

    m_a = marche_aleatoire()
    monte_carlo = monte_carlo()

    # Simulation loop
    for i in range(monte_carlo.N):
        temp = monte_carlo.stoc_walk(monte_carlo.s0,monte_carlo.drift,monte_carlo.volatility,monte_carlo.days)
        if temp > monte_carlo.k:
            payoff = temp-monte_carlo.k
            payoff = payoff*np.exp(-monte_carlo.r/monte_carlo.t_*monte_carlo.days)
            monte_carlo.avg += payoff
        else:
            monte_carlo.zero_trials += 1

    # Averaging the payoffs
    price = monte_carlo.avg/float(monte_carlo.N)

    # Priting the results
    print("MONTE CARLO PLAIN VANILLA CALL OPTION PRICING")
    print("Option price: ",price)
    print("Initial price: ",monte_carlo.s0)
    print("Strike price: ",monte_carlo.k)
    print("Daily expected drift: ",monte_carlo.drift*100,"%")
    print("Daily expected volatility: ",monte_carlo.volatility*100,"%")
    print("Total trials: ",monte_carlo.N)
    print("Zero trials: ",monte_carlo.zero_trials)
    print("Percentage of total trials: ",monte_carlo.zero_trials/monte_carlo.N*100,"%")   

    # Marches aleatoires d'Apple générées d'ajd à date de maturité de l'option
    last_value_walk = marche_aleatoire.last_value_walk
    data = marche_aleatoire.data
    y_aapl = marche_aleatoire.y_aapl
    m_a.generate_aapl_market(y_aapl, data)
    m_a.affichage_valhist_marche(y_aapl, data, last_value_walk)

    # Calcul du payoff pour chaque marche aleatoire
    payoff_marche = monte_carlo.long_call(last_value_walk, monte_carlo.k)
    print("Payoff pour chaque marche générée: ",payoff_marche)
    # Moyenne des payoffs calculés
    print("Moyenne des payoffs calculés: ",mean(payoff_marche))

    # Calcul du prix de l'option avec le modèle Black Scholes
    C = monte_carlo.call(monte_carlo.s0, monte_carlo.k, monte_carlo.T, monte_carlo.r, monte_carlo.drift, monte_carlo.volatility)
    print("Prix de l'option avec Black Scholes: ", C)


