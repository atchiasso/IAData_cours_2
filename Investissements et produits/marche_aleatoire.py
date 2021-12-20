import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from datetime import date
from datetime import timedelta, date
from yahoofinancials import YahooFinancials
from random import randint
import pandas_datareader as web
import numpy as np

class marche_aleatoire:

    ticker = "AAPL"
    data = pd.DataFrame()
    data[ticker] = web.DataReader(ticker, data_source = 'yahoo', 
                                start = "2020-01-01", 
                                end = "today")['Adj Close']
    data = data.dropna()

    last_value_walk = []
    y_aapl = []
    price_list = data.values.tolist()
    
    for sublist in price_list:
        for val in sublist:
            y_aapl.append(val)
    
    # Définition d'une marche aléatoire
    def marche(self, valprec):
        if(valprec == 0):
            return valprec+1
        value = randint(valprec-1,valprec+1)
        while(value == valprec):
            value = self.marche(valprec)
        return value

    # Initialisation des marches aléatoires (nombre de marches à définir)
    def initialize_marche(self, nb, debut=0):
        list_market = []
        list_market.append(debut)
        
        for i in range(nb):
            value = self.marche(list_market[i])
            if(value < 0):
                list_market.append(0)
            else:
                list_market.append(value)
        return list_market

    # Initialisation de 1000 marches aléatoires
    def initialize_marche2(self, debut=0):
        list_market = []
        list_market.append(debut)
        
        for i in range(999):
            value = self.marche(list_market[i])
            if(value < 0):
                list_market.append(0)
            else:
                list_market.append(value)
        return list_market

    # Fournit une liste de date
    def daterange(self, start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)

    # Génération d'une liste de 100 dates
    def generate_list_date(self, start_date, end_date):
        list_date = []
        i = 0

        for single_date in self.daterange(start_date, end_date):
            i += 1
            list_date.append(single_date.strftime("%Y-%m-%d"))
            if(i == 100):
                return list_date

    # Génération d'une liste de 1000 dates
    def generate_list_date2(self, start_date, end_date):
        list_date = []
        i = 0

        for single_date in self.daterange(start_date, end_date):
            i += 1
            list_date.append(single_date.strftime("%Y-%m-%d"))
            if(i == 1000):
                return list_date

    # Affichage des 100 marches aleatoires
    def affichage_100_marches_aleatoires(self):
        x = self.generate_list_date(date(2021, 3, 1), date.today())
        y = self.initialize_marche(99)
        plt.figure(1, figsize=(5, 3))
        plt.plot(x,y)
        plt.show()

    # Valeurs historiques AAPL 
    def affichage_val_hist_appl(self, data):
        plt.figure(1, figsize=(5, 3))
        plt.plot(data)
        plt.show()

    # Génération des valeurs historiques AAPL
    def generate_aapl_market(self, y_aapl, data):
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days = 1)
        tomorrow = today + datetime.timedelta(days = 1)

        x_generate = self.generate_list_date2(tomorrow, date(2030, 1, 1))
        axe_x = data.index.tolist() + x_generate
        y_generate = self.initialize_marche2(int(y_aapl[-1]))
        axe_y = y_aapl + y_generate
        return axe_x, axe_y

    # Affichage des valeurs historiques de la marche aléatoire
    def affichage_valhist_marche(self, y_aapl, data, last_value_walk):
        color_list = ['b-', 'g-', 'r-', 'c-', 'v-']
        for i in range(5):
            axe_x, axe_y = self.generate_aapl_market(y_aapl, data)
            plt.figure(1, figsize=(50, 30))
            plt.plot(axe_x,axe_y, color_list[i])
            last_value_walk.append(axe_y[-1])
        print(last_value_walk)
        plt.show()

if __name__ == '__main__':

    m_a = marche_aleatoire()
    
    # Log Return
    log_returns = np.log(1 + m_a.data.pct_change())
    # Affichage de la fin du tableau
    log_returns.tail()

    # Variance 
    weight = np.array([0.25])
    pfolio_var = round(np.dot(weight.T, np.dot(log_returns.cov()*250, weight)),2)
    print("La variance du portefeuille est de " + str(pfolio_var*100) + str("%"))

    # Affichage des 100 marches aleatoires
    m_a.generate_list_date(start_date = date(2021, 3, 1), end_date = date.today())
    m_a.affichage_100_marches_aleatoires()

    # Affichage valeurs historiques AAPL
    m_a.affichage_val_hist_appl(m_a.data)

    # Affichage valeurs historiques AAPL + marches aléatoires à partir de la dernière date de valeurs historiques
    m_a.generate_aapl_market(m_a.y_aapl, m_a.data)
    m_a.affichage_valhist_marche(m_a.y_aapl, m_a.data, m_a.last_value_walk)