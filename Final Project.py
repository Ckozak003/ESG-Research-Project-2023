#!/usr/bin/env python
# coding: utf-8

# # Appendix

# In[288]:


import numpy as np
import yfinance as yf
import pandas as pd
import os
import requests
import pandoc


# In[287]:


get_ipython().system('pip3 install yesg')
get_ipython().system('pip3 install pandoc')
import yesg


# In[7]:


SP_500 = pd.read_csv("constituents_csv.csv")
#Reading in ticker info


# In[103]:


tickers = SP_500['Symbol']
Ratings = pd.DataFrame(SP_500['Name'])
Ratings['Symbol'] = SP_500.Symbol
dataframes = []

for i in tickers:
    try:
        df = yesg.get_esg_short(i)
        dataframes.append(df)
    except: continue
Ratings = pd.concat(dataframes, ignore_index = True)
Ratings.reindex
Ratings.head(10)
#Seeing which stocks on the S&P500 have data for ESG ratings and pulling them into one dataframe


# In[269]:


Ratings


# In[110]:


Ratings2 = Ratings.astype({'Total-Score': 'float', 'E-Score': 'float',
                           'S-Score': 'float', 'G-Score': 'float'})
Ratings2 = Ratings2.sort_values('Total-Score', ascending = True)
Ratings2 = Ratings2.reset_index()
Bin1 = pd.DataFrame(Ratings2.iloc[:143])
Bin2 = pd.DataFrame(Ratings2.iloc[144:287])
Bin3 = pd.DataFrame(Ratings2.iloc[288:431])
#Reformatting meta detaframe, sorting by ascending values, splitting large dataset into smaller chunks


# In[270]:


a = Ratings2.describe()
x = Bin1.describe()
y = Bin2.describe()
z = Bin3.describe()
display(a,x,y,z)
#getting basic descriptive stats for each bin


# In[129]:


Bin1.loc[Bin1['Total-Score'] == 13.9]     #find stocks around median for ESG ratings for each bin
port1 = pd.DataFrame(Bin1.iloc[66:77])
tickers1 = port1.Ticker.values.tolist()
port1


# In[271]:


Bin2.loc[Bin2['Total-Score'] == 21]
port2 = pd.DataFrame(Bin2.iloc[66:77])
tickers2 = port2.Ticker.values.tolist()
port2


# In[280]:


port3 = pd.DataFrame(Bin3.iloc[66:77])
tickers3 = port3.Ticker.values.tolist()
port3


# In[236]:


#Lets start building an efficient frontier based on bins
tickers_df1 = yf.download(tickers1, 
                      start='2022-01-01', 
                      end='2023-04-02', 
                      progress=False, auto_adjust=True)

from math import log

returns = tickers_df1['Close'].applymap(log).diff()[1:]
prices = tickers_df1['Close']
stats = returns.agg(['mean', 'std', 'var'])
correl = returns.corr()
display(stats, correl)


# In[272]:


tickers_df2 = yf.download(tickers2, 
                      start='2022-01-01', 
                      end='2023-04-02', 
                      progress=False, auto_adjust=True)

from math import log

returns2 = tickers_df2['Close'].applymap(log).diff()[1:]
prices2 = tickers_df2['Close']
stats2 = returns2.agg(['mean', 'std', 'var'])
correl2 = returns2.corr()
display(stats2, correl2)


# In[281]:


tickers_df3 = yf.download(tickers3, 
                      start='2022-01-01', 
                      end='2023-04-02', 
                      progress=False, auto_adjust=True)

from math import log

returns3 = tickers_df3['Close'].applymap(log).diff()[1:]

stats3 = returns3.agg(['mean', 'std', 'var'])
correl3 = returns3.corr()
display(stats3, correl3)


# In[282]:


#Annualizing data

#Port1

annual_returns1 = stats.transpose()['mean'] * 252
annual_covar1 = returns.cov() * 252

#Port2

annual_returns2 = stats2.transpose()['mean'] * 252
annual_covar2 = returns2.cov() * 252

#Port3

annual_returns3 = stats3.transpose()['mean'] * 252
annual_covar3 = returns3.cov() * 252


display(annual_covar1,annual_covar2,annual_covar3)
display(annual_returns1,annual_returns2,annual_returns3)


# In[154]:


import sys
get_ipython().system('{sys.executable} --version')
# !{sys.executable} -m pip install cvxpy 
# !{sys.executable} -m pip install PyPortfolioOpt

import math
import numpy as np
from datetime import datetime
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')


# In[198]:


#Used from Quiz 1 Answer Sheet

from numpy import array, dot
from qpsolvers import solve_qp

class create_efficient_frontier():
    def __init__(self, returns, covar):
        self.returns = np.array(returns)
        self.covar = np.array(covar)
        self.n =len(self.covar)
        self.tickers = list(returns.index)

    def get_portfolio(self, return_target):
        """for a given target return create lowest variance long-only portfolio"""
        P, q = self.covar, np.array([0.] * self.n)
        G, h = None,None 
        A = np.array([annual_returns, np.array([1.0] * self.n)])
        b = np.array([return_target, 1.0])
        lb, ub = np.array([0.] * self.n),np.array([1.] * self.n)

        self.portfolio = solve_qp(P, q, G, h, A, b, lb, ub,solver='osqp')
        return {"portfolio":self.portfolio, "risk_ret":self.risk_return()}  # return (allocation , risk-return)

    def risk_return(self):
        """return the risk and return for this portfolio"""
        return np.sqrt(self.portfolio.dot(self.covar.dot(self.portfolio))),\
            self.returns.dot(self.portfolio)


# In[250]:


#EF from best best ESG portfolio
import math
annual_returns = annual_returns1
annual_covar = annual_covar1
ef = create_efficient_frontier(annual_returns, annual_covar)
min_return, max_return = min(annual_returns) + .001, max(annual_returns) - .001
frontier = np.array([ef.get_portfolio(r)['risk_ret'] 
                     for r in np.linspace(min_return, max_return, 20)]).T

# plot the efficient frontier in the Markowitz risk-return space

plt.plot(frontier[0], frontier[1], 'o-', color='blue') # plot the efficient frontier
for n, r, s in zip(annual_returns1.index, annual_returns1, np.sqrt(np.diag(annual_covar1))):
    plt.plot([s], [r], 'o', color='red')
    plt.text(s+0.005, r, n)
plt.title('Efficient Frontier of Portfolio 1')
plt.xlabel('Risk', fontsize=20)
plt.ylabel('Return', fontsize=20)
plt.show()


# In[251]:


#Looking at allocations

min_var_risk = min(frontier[0])
min_var_portfoio_number = [i for i, risk in enumerate(frontier[0]) if risk == min_var_risk][0]

print("min variance portfolio:")
print(f"efficient frontier portfolio # = {min_var_portfoio_number}")

print(f"(risk, return) = ({frontier[0][min_var_portfoio_number]:0.4f}, {frontier[1][min_var_portfoio_number]:0.4f})")

frontier_portfolios = np.array([ef.get_portfolio(r)['portfolio'] 
                     for r in np.linspace(min_return, max_return, 20)])

ticker_list = list(annual_returns.index)
allocation1 = pd.DataFrame(frontier_portfolios[min_var_portfoio_number], index=ticker_list, columns=['allocation'])
display(allocation1)


# In[274]:


#EF from best middle ESG portfolio
import math
annual_returns = annual_returns2
annual_covar = annual_covar2
ef = create_efficient_frontier(annual_returns, annual_covar)
min_return, max_return = min(annual_returns), max(annual_returns)
frontier = np.array([ef.get_portfolio(r)['risk_ret'] 
                     for r in np.linspace(min_return, max_return, 20)]).T

# plot the efficient frontier in the Markowitz risk-return space

plt.plot(frontier[0], frontier[1], 'o-', color='blue') # plot the efficient frontier
for n, r, s in zip(annual_returns1.index, annual_returns, np.sqrt(np.diag(annual_covar))):
    plt.plot([s], [r], 'o', color='red')
    plt.text(s+0.005, r, n)
plt.title('Efficient Frontier of Portfolio 2')
plt.xlabel('Risk', fontsize=20)
plt.ylabel('Return', fontsize=20)
plt.show()


# In[275]:


min_var_risk = min(frontier[0])
min_var_portfoio_number = [i for i, risk in enumerate(frontier[0]) if risk == min_var_risk][0]

print("min variance portfolio:")
print(f"efficient frontier portfolio # = {min_var_portfoio_number}")

print(f"(risk, return) = ({frontier[0][min_var_portfoio_number]:0.4f}, {frontier[1][min_var_portfoio_number]:0.4f})")

frontier_portfolios = np.array([ef.get_portfolio(r)['portfolio'] 
                     for r in np.linspace(min_return, max_return, 20)])

ticker_list = list(annual_returns.index)
allocation2 = pd.DataFrame(frontier_portfolios[min_var_portfoio_number], index=ticker_list, columns=['allocation'])
display(allocation2)


# In[283]:


#EF from best worst ESG portfolio including SVB
import math
annual_returns = annual_returns3
annual_covar = annual_covar3
ef = create_efficient_frontier(annual_returns, annual_covar)
min_return, max_return = min(annual_returns), max(annual_returns)
frontier = np.array([ef.get_portfolio(r)['risk_ret'] 
                     for r in np.linspace(min_return, max_return, 15)]).T

# plot the efficient frontier in the Markowitz risk-return space

plt.plot(frontier[0], frontier[1], 'o-', color='blue') # plot the efficient frontier
for n, r, s in zip(annual_returns.index, annual_returns, np.sqrt(np.diag(annual_covar))):
    plt.plot([s], [r], 'o', color='red')
    plt.text(s+0.005, r, n)
plt.title('Efficient Frontier of Portfolio 3')
plt.xlabel('Risk', fontsize=20)
plt.ylabel('Return', fontsize=20)
plt.show()

#this includes SVB


# In[284]:


#Removing SVB and recalculating returns, vol, etc

SVB_ind = port3.index[(port3['Ticker'] == 'SIVB')]
port3 = port3.drop(SVB_ind)
port3

tickers3 = port3.Ticker.values.tolist()


tickers_df3 = yf.download(tickers3, 
                      start='2022-01-01', 
                      end='2023-04-02', 
                      progress=False, auto_adjust=True)

from math import log

returns3 = tickers_df3['Close'].applymap(log).diff()[1:]
prices3 = tickers_df3['Close']
stats3 = returns3.agg(['mean', 'std', 'var'])
correl3 = returns3.corr()
display(stats3, correl3)

annual_returns3 = stats3.transpose()['mean'] * 252
annual_covar3 = returns3.cov() * 252


# In[286]:


#EF from worst ESG portfolio without SVB
import math
annual_returns = annual_returns3
annual_covar = annual_covar3
ef = create_efficient_frontier(annual_returns, annual_covar)
min_return, max_return = min(annual_returns), max(annual_returns)
frontier = np.array([ef.get_portfolio(r)['risk_ret'] 
                     for r in np.linspace(min_return, max_return, 20)]).T

# plot the efficient frontier in the Markowitz risk-return space

plt.plot(frontier[0], frontier[1], 'o-', color='blue') # plot the efficient frontier
for n, r, s in zip(annual_returns.index, annual_returns, np.sqrt(np.diag(annual_covar))):
    plt.plot([s], [r], 'o', color='red')
    plt.text(s+0.005, r, n)
plt.title('Efficient Frontier of Portfolio 3')
plt.xlabel('Risk', fontsize=20)
plt.ylabel('Return', fontsize=20)
plt.show()

#this DOES NOT includes SVB


# In[256]:


min_var_risk = min(frontier[0])
min_var_portfoio_number = [i for i, risk in enumerate(frontier[0]) if risk == min_var_risk][0]

print("min variance portfolio:")
print(f"efficient frontier portfolio # = {min_var_portfoio_number}")

print(f"(risk, return) = ({frontier[0][min_var_portfoio_number]:0.4f}, {frontier[1][min_var_portfoio_number]:0.4f})")

frontier_portfolios = np.array([ef.get_portfolio(r)['portfolio'] 
                     for r in np.linspace(min_return, max_return, 20)])

ticker_list = list(annual_returns.index)
allocation3 = pd.DataFrame(frontier_portfolios[min_var_portfoio_number], index=ticker_list, columns=['allocation'])
display(allocation3)


# In[267]:


### For Portfolio 1 (Best ESG)


### We use the minimum variance portfolio to calculate these VaR's
returns_p1 = prices.resample('D').last().pct_change()
returns_p1 = returns_p1[1:]
display(returns_p1)
p1_returns = returns_p1.dot(allocation1)

var_level = 0.05
portfolio_var1 = np.percentile(p1_returns, var_level*100)

plt.hist(p1_returns, bins=20)
plt.axvline(x=portfolio_var1, color='r', linestyle='-')
plt.title("Disributions of Returns and 95% Var Portfolio 1")
plt.ylabel('Frequency')
plt.xlabel('Returns')
plt.show()

print("The 95% VaR for porfolio 1 in terms of returns is: ", portfolio_var1)


# In[265]:


### For Portfolio 2 (Middle ESG)


### We use the minimum variance portfolio to calculate these VaR's
returns_p2 = prices2.resample('D').last().pct_change()
returns_p2 = returns_p2[1:]
display(returns_p2)
p2_returns = returns_p2.dot(allocation2)

var_level = 0.05
portfolio_var2 = np.percentile(p2_returns, var_level*100)

plt.hist(p2_returns, bins=20)
plt.axvline(x=portfolio_var2, color='r', linestyle='-')
plt.title("Disributions of Returns and 95% Var Portfolio 2")
plt.ylabel('Frequency')
plt.xlabel('Returns')
plt.show()

print("The 95% VaR for porfolio 2 in terms of returns is: ", portfolio_var2)


# In[266]:


### For Portfolio 3 (Worst ESG)


### We use the minimum variance portfolio to calculate these VaR's
returns_p3 = prices3.resample('D').last().pct_change()
returns_p3 = returns_p3[1:]
display(returns_p3)
p3_returns = returns_p3.dot(allocation3)

var_level = 0.05
portfolio_var3 = np.percentile(p3_returns, var_level*100)

plt.hist(p3_returns, bins=20)
plt.axvline(x=portfolio_var3, color='r', linestyle='-')
plt.title("Disributions of Returns and 95% Var Portfolio 3")
plt.ylabel('Frequency')
plt.xlabel('Returns')
plt.show()

print("The 95% VaR for porfolio 3 in terms of returns is: ", portfolio_var3)


# In[ ]:


plt.plot(frontier[0], frontier[1], 'o-', color='blue') # plot the efficient frontier
for n, r, s in zip(annual_returns1.index, annual_returns1, np.sqrt(np.diag(annual_covar1))):
    plt.plot([s], [r], 'o', color='red')
    plt.text(s+0.005, r, n)

plt.title('Efficient Frontier of Portfolio 1')
plt.xlabel('Risk', fontsize=20)
plt.ylabel('Return', fontsize=20)

