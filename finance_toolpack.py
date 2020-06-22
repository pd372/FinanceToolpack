
"""
Created on Fri May 29 15:05:59 2020

@author: Pedro D'Avila

Collection of functions based on Yahoo Finance APIs. The goal here is to assist on portfolio management
and finacial analysis.


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import mplfinance as mpf
import datetime as dt
from datetime import date
from scipy.stats import norm
from matplotlib import style
from dateutil import parser
from scipy import stats
import statsmodels.api as sm
import os
from urllib.request import urlopen
import json
import yfinance as yf
import yahoo_fin.stock_info as si




#creates portfolio with a list of securities
def createPortfolio(securities, start=False):
    df=pd.DataFrame()
    if start:
        for security in securities :
            df[security]=web.DataReader(security, 'yahoo', start = start)['Adj Close']
        df.dropna(0,inplace=True)  
    else:
        for security in securities :
            df[security]=web.DataReader(security, 'yahoo', start='2015-1-1')['Adj Close']
        df.dropna(0,inplace=True)
    return df
  
    
  
#normalizing to 100
def normalize (df):
    normalized=(df/df.iloc[0])*100
    return normalized


def logRtns(df):
    log_rtns=np.log(df/df.shift(1))
    log_rtns.fillna(0,inplace=True)
    return log_rtns


def annualLogRtns(df):
    rtns=logRtns(df).mean()*250
    return rtns


def getAnnualizedStdev(ticker):
    df=web.DataReader(ticker,'yahoo',start='2005')['Adj Close']
    stdev=logRtns(df).std()*250**0.5
    return stdev
    
    

#annualized covariance matrix
def getCovMatrix(df):
    covMatrix=logRtns(df).cov()*250
    return covMatrix



def getCorrMatrix(df):
    corrMatrix=logRtns(df).corr()
    return corrMatrix


def assignRandomWeights(df):
    num_assets=len(df.columns)
    weights = np.random.random(num_assets) #generates num_assets random float points in an array
    #Now we have to make the random numbers sum to 100%
    weights/= np.sum(weights)
    return weights



#Expected portfolio return ---> np.sum() adds the elemetns inside the same array. np.add() adds 2 arrays
def portExpectedRtn(df, weights):
    Er=np.sum(weights * annualLogRtns(df))
    return Er# we have to add the probability of a recession here!

#Historical Performance given number of shares
def portHistPerformance(df, nShares):
    nShares=np.array(nShares)
    rtn= pd.DataFrame(index=df.index.values)
    df=df.values*nShares
    df=df.sum(axis=1)
    rtn['Historical Performance'] = df
    return rtn

#individual total returns during holding period
def portIndivRtns(df):
    df=logRtns(df).sum(axis=0)
    return df


#Expected portfolio Variance   
def portExpectedVar(df,weights):
    #Expected portfolio Variance
    Evar=np.dot(weights.T,np.dot( getCovMatrix(df), weights))
    return Evar



#Expected portfolio volatility -> remember to use the np methods for array operations
def portExpectedVolatility(df, weights):
    Evol=np.sqrt(portExpectedVar(df, weights))
    return Evol


#get the dividend yield of the portfolio
def portDivYield(df, nShares):
    
    dvds=[]
    nShares=np.array(nShares)
    
    for sec in df.columns.values:
        stts=si.get_stats(sec).set_index(['Attribute'])
        div=stts.loc['Trailing Annual Dividend Rate 3'].values[0]
        div=float(div)
            
        dvds.append(div)
    dvds=np.array(dvds)
    dvds=np.nan_to_num(dvds)
    totDvds=np.sum(dvds*nShares)
    totInvested=np.sum((df.iloc[-1]*nShares).values)
    
    yld=totDvds/totInvested
    
    return yld



#Running a simulation of 1000 different portfolio weights to create the Efficient Frontier
def getEfficientFrontier(df):
    pfolioRtns=[]
    pfolioVolats=[]
    for w in range(1000):
        weights=assignRandomWeights(df)
        pfolioRtns.append(portExpectedRtn(df, weights))
        pfolioVolats.append(portExpectedVolatility(df,weights))
        
    pfolioRtns=np.array(pfolioRtns)
    pfolioVolats=np.array(pfolioVolats)
    
    ptf= pd.DataFrame({'Return':pfolioRtns, 'Volatility':pfolioVolats})
    ptf.plot(x='Volatility', y='Return', kind='scatter',
             title='Markowitz Efficient Frontier:\n')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Return')
    plt.show()
 
    

def getStockBeta(ticker, years):
    #getting 'years' years of data
    today=date.today()
    td=str(today)
    tdYr=int(td[:4])
    startYr=tdYr-years
    startDate='{}-01-01'.format(startYr)
    
    df=pd.DataFrame()
    assets=[ticker, '^GSPC']
    
    for asset in assets:
        df[asset]=web.DataReader(asset,'yahoo', start=startDate)['Adj Close']
        
    covMtx=getCovMatrix(df)
    mktCov=covMtx.iloc[0,1]
    mktVar=covMtx.iloc[1,1]
    beta=mktCov/mktVar
    return beta



def getRiskFreeRate(): 
    today=date.today()
    td=str(today)
    curYr=int(td[:4])
    curMo=int(td[5:7])
    startDay=1
    
    if (curMo-2)<=0:
        curMo=11 
        curYr -= 1
    else:
        curMo=curMo - 2 
    startDate='{}-{}-{}'.format(curYr,curMo,startDay)
    #T-bill yield
    Rf=float(web.DataReader('TB1YR', 'fred', start=startDate).iloc[-1])/100
    return Rf



def getPortBeta(df, weights):
    betas=[]
    for sec in df.columns:
        b=getStockBeta(sec, 5)
        betas.append(b)
    betas=np.array(betas)
    B=np.sum(betas*weights)
    return B


#Adjusted returns for beta and the recession probability
def portExpectedRtnAdj(df, weights):
    today=date.today()
    td=str(today)
    curYr=int(td[:4])   
    data = web.DataReader('RECPROUSM156N', 'fred', start='{}-1-1'.format(curYr))
    recProb=float(data.iloc[-1]/100)
    Er=np.sum(weights * annualLogRtns(df)*(1-(getPortBeta(df,weights)*recProb)))
    return Er
    

def getCAPM(ticker):
    Rf=getRiskFreeRate()
    beta=getStockBeta(ticker,5)
    df=web.DataReader('^GSPC','yahoo',start='2005-1-1')['Adj Close']
    Rm=annualLogRtns(df)
    Ri=Rf+beta*(Rm-Rf)
    return Ri


#CAPM with the estimated market return being 6%
def getFastCAPM(ticker):
    Rf=getRiskFreeRate()
    beta=getStockBeta(ticker,5)
    Rm=0.06
    Ri=Rf+beta*(Rm-Rf)
    return Ri
    

#Get Sharpe Ratio from a security or portfolio since 2005
def stockSharpe(ticker):
    Er=getCAPM(ticker)
    Rf=getRiskFreeRate()
    stdev=getAnnualizedStdev(ticker)
    sharpe=(Er-Rf)/stdev
    return sharpe


def portSharpe(df,weights):
    Er=portExpectedRtn(df,weights)
    Rf=getRiskFreeRate()
    stdev=portExpectedVolatility(df,weights)
    sharpe=(Er-Rf)/stdev
    return sharpe
        
def sharpeAssist(df, weights, Rf):
    Er=portExpectedRtn(df,weights)
    stdev=portExpectedVolatility(df,weights)
    sharpe=(Er-Rf)/stdev
    return sharpe
    
#run a simulation to get the points in the efficient frontier with the highest sharpe, and the corresponfing weights
def maximizeSharpeRatio(df):  
    #dictionary with sharpes and corresponding weights
    sharpes={}
    mxSharpe=0
    Rf=getRiskFreeRate()
    
    for w in range(1000):
        weights = assignRandomWeights(df)
        sharpe=sharpeAssist(df, weights, Rf)
        sharpes[sharpe]=weights
        if sharpe> mxSharpe:
            mxSharpe=sharpe
    rt= pd.Series(sharpes[mxSharpe], index=df.columns)
    rt.loc['Max Sharpe:']=mxSharpe
    rt.loc['Return:']=portExpectedRtn(df,sharpes[mxSharpe])
    rt.loc['Risk:']=portExpectedVolatility(df, sharpes[mxSharpe])
    return rt




def maximizeRtn(df):
    rtns={}
    mxRtn=0

    for w in range(1000):
        weights = assignRandomWeights(df)
        rtn=portExpectedRtn(df,weights)
        rtns[rtn]=weights
        if rtn> mxRtn:
            mxRtn=rtn
    rt= pd.Series(rtns[mxRtn], index=df.columns)
    rt.loc['Return:']=mxRtn
    rt.loc['Risk:']=portExpectedVolatility(df, rtns[mxRtn])
    return rt
    
    
def minimizeRisk(df):
    rsks={}
    minRsk=9999999

    for w in range(1000):
        weights = assignRandomWeights(df)
        rsk=portExpectedVolatility(df,weights)
        rsks[rsk]=weights
        if rsk < minRsk:
            minRsk=rsk
    rt= pd.Series(rsks[minRsk], index=df.columns)
    rt.loc['Return:']=portExpectedRtn(df, rsks[minRsk])
    rt.loc['Risk:']=minRsk
    return rt

        
#Returns a chart of 10 possible paths for stock price in a 1000 day future period         
def getMonteCarlo(ticker):
    #Po=P1 * e**(drift+Stdev*Z(Random[0:1]))
    df = pd.DataFrame()
    df[ticker]=web.DataReader(ticker, 'yahoo',start='2010-01-01')['Adj Close']
    lg=logRtns(df)
    u=lg.mean()
    var=lg.var()
    drift=(u-0.5*var)
    #---------
    stdev=lg.std()
    #We can easily transform a pandas series into an array by doing df.values or call np.array method
    drift=drift.values
    stdev=stdev.values
    #Scenarios and futureRange
    futRange=1000
    scenarios=10
    #------------
    Z=norm.ppf(np.random.rand(futRange, scenarios))
    
    #r = drift + Volatility <=> drift+(stdev*Z)
    r=drift+stdev*Z
    
    #Now we calculate e**r
    E_r= np.exp(r)
    
    #Now we get the last price data available 
    Po = df.iloc[-1]
    
    #Now we have to create an array as big as the one from the future returns. the method is creating it and filling w/ 0s 
    priceLst=np.zeros_like(E_r)
    
    #Remember Pt=Pt-1*e**r
    priceLst[0]=Po
    #Now we loop through all the days, starting from day 1 (since 0 is aready filled) and use the formula above
    for t in range(1,futRange):
        priceLst[t]=priceLst[t-1]*E_r[t]
    #plotting  
    style.use('fivethirtyeight')
    plt.plot(priceLst)
    plt.title(ticker+': Future Scenarios of Expected Price \n Brownian Motion')
    plt.ylabel('Expected Price')
    plt.xlabel('Days in the future')
    plt.show()
    return priceLst



def getCorrelationHeatmap (df):
    
    corr_table = getCorrMatrix(df)
    
    data= corr_table.values# this gives us just the inner values of the table, no headers or index
    fig=plt.figure()
    
    ax=fig.add_subplot(1,1,1) 
    
    #this is how we create a heatmap, we use the color scale and ticks 
    heatmap=ax.pcolor(data,cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)  
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    column_labels=corr_table.columns
    row_labels= corr_table.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    
    plt.xticks(rotation=90)
    
    heatmap.set_clim(-1,1)#sets the limit of the color scale since it is a correlation matrix
    
    plt.tight_layout()
    plt.show()
 
    

def visualize(ticker, smav=False):
    df = pd.DataFrame()
    df= web.DataReader(ticker,'yahoo',start='2000-1-1')
    if smav:
        a= int(input('Enter smav1:'))
        b=int(input('Enter smav2:'))
        mpf.plot(df,style='yahoo', volume=True, mav=(a,b),title=ticker, ylabel='Price',ylabel_lower='Volume') 
    else:
        mpf.plot(df,style='yahoo', volume=True,title=ticker, ylabel='Price',ylabel_lower='Volume')



#Black Scholes option pricing
#S: price, K: strike price, T: maturity date
def d1(S,K,r,stdev, T):
    return (np.log(S/K) + (r + stdev**2 / 2) *T) / (stdev*np.sqrt(T))

def d2(S,K,r,stdev, T):
    return (np.log(S/K) + (r - stdev**2 / 2) *T) / (stdev*np.sqrt(T))

#get the price of a call option with black scholes merton 
def getBSMCall(ticker, K ,T):
    df=web.DataReader(ticker,'yahoo',start='2010-1-1')['Adj Close']
    #stock price
    S=df.iloc[-1]
    #risk-free rate
    r=getRiskFreeRate()
    #stdev
    stdev=getAnnualizedStdev(ticker)
    #get time horizon in years
    convT=parser.parse(T).date()
    today=date.today()
    delta=convT-today
    T=delta.days/365
    return(S*norm.cdf(d1(S,K,r,stdev,T)))-(K*np.exp(-r*T)*norm.cdf(d2(S,K,r,stdev,T)))




def getCallPrice(ticker, K, T):
    df=web.DataReader(ticker,'yahoo',start='2010-1-1')['Adj Close']
    r=getRiskFreeRate()
    stdev=getAnnualizedStdev(ticker)
    stdev=np.array([stdev])
    
    #Time horizon
    convT=parser.parse(T).date()
    today=date.today()
    delta=convT-today
    T=delta.days/365
    tIntervals=250
    deltaT=T/tIntervals
    scenarios=1000
    Z=np.random.standard_normal((tIntervals+1,scenarios))
    S=np.zeros_like(Z)
    Po=df.iloc[-1]
    S[0]=Po
    
    for t in range(1, tIntervals+1):
        S[t]=S[t - 1] * np.exp((r - 0.5 * stdev**2) * deltaT * stdev * deltaT**0.5 * Z[t])
        
    payOff=np.maximum(S[-1]-K,0)
    C=(np.exp(-r*T)*np.sum(payOff)) / scenarios
    return C





#regression Analysis
def runOLSRegression(X,Y):
    X=np.array(X,dtype='float64')
    Y=np.array(Y,dtype='float64')
    X1=sm.add_constant(X)
    regression=sm.OLS(Y,X1).fit()
    #plot if it is a simple regression (only one column for independent variable)
    try:
        np.shape(X)[1]==None
    except IndexError:
        slope,intercept, r_value, p_value, std_err=stats.linregress(X,Y)
        y = intercept + slope*X
        plt.plot(X, y, '-r', label= '{}={}+{}*{}'.format(Y.name,round(intercept,2),round(slope,2),X.name))
        plt.scatter(X,Y)
        plt.title('Regression:\n'+X.name + ' & '+Y.name)
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()
    return regression.summary()
        
         
      
        
def getStockData(ticker, timeFrame):
    today=date.today()
    yr=int(str(today)[:4])
    tgt=yr-timeFrame
    stt='{}-1-1'.format(tgt)
    df=web.DataReader(ticker, 'yahoo',start=stt)
    return df



def getDescription(ticker):
    comp=yf.Ticker(ticker)
    des=comp.info['longBusinessSummary']
    return des


    






    




