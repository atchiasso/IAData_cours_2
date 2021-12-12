import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import pandas_datareader as web
import numpy as np
import json
from datetime import date
from datetime import timedelta, date
from yahoofinancials import YahooFinancials
from random import randint
from marche_aleatoire import generate_list_date, affichage_100_marches_aleatoires


class monte_carlo_pricing():
    # Stocastic walk
    # This function calculates the stochastic integral after periods
    # and returns the final price.
    def stoc_walk(self, p, dr, vol, periods):
        w = np.random.normal(0,1,size=periods)
        for i in range(periods):
            p += dr*p + w[i]*vol*p
        return p



if __name__ == '__main__':

    monte_carlo = monte_carlo_pricing()
    symbol = "AAPL"

    # Create the object representing the symbol of interest
    equity = yf.Ticker(symbol)

    current_price = equity.info['currentPrice']
    print(f"Current price of {symbol} - ${current_price}")

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

    # Simulation loop
    for i in range(N):
        temp = monte_carlo.stoc_walk(s0,drift,volatility,days)
        if temp > k:
            payoff = temp-k
            payoff = payoff*np.exp(-r/t_*days)
            avg += payoff
        else:
            zero_trials += 1

    # Averaging the payoffs
    price = avg/float(N)

    # Priting the results
    print("MONTE CARLO PLAIN VANILLA CALL OPTION PRICING")
    print("Option price: ",price)
    print("Initial price: ",s0)
    print("Strike price: ",k)
    print("Daily expected drift: ",drift*100,"%")
    print("Daily expected volatility: ",volatility*100,"%")
    print("Total trials: ",N)
    print("Zero trials: ",zero_trials)
    print("Percentage of total trials: ",zero_trials/N*100,"%")

