import traceback
import json
import imp
import urllib
import urllib2
import webbrowser
import re
import datetime
import time
import inspect
import os
import os.path
import sys
import ssl
from copy import deepcopy

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style
import matplotlib.pyplot as plt

import Tkinter as tk
import ttk


def loadData(marketList=None, dataToLoad=None, refresh=False, beginInSample=None, endInSample=None, dataDir = 'tickerData'):
    ''' prepares and returns market data for specified markets.

        prepares and returns related to the entries in the dataToLoad list. When refresh is true, data is updated from the Quantiacs server. If inSample is left as none, all available data dates will be returned.

        Args:
            marketList (list): list of market data to be supplied
            dataToLoad (list): list of financial data types to load
            refresh (bool): boolean value determining whether or not to update the local data from the Quantiacs server.
            beginInSample (str): a str in the format of YYYYMMDD defining the begining of the time series
            endInSample (str): a str in the format of YYYYMMDD defining the end of the time series

        Returns:
            dataDict (dict): mapping all data types requested by dataToLoad. The data is returned as a numpy array or list and is ordered by marketList along columns and date along the row.

    Copyright Quantiacs LLC - March 2015
    '''
    if marketList is None:
        print "warning: no markets supplied"
        return

    dataToLoad = set(dataToLoad)
    requiredData = set(['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'P', 'RINFO',
                        'p'])

    dataToLoad.update(requiredData)

    nMarkets = len(marketList)

    # set up data director
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)

    for j in range(nMarkets):
        path = os.path.join(dataDir, marketList[j]+'.txt')

        # check to see if market data is present. If not (or refresh is true), download data from quantiacs.
        if not os.path.isfile(path) or refresh:
            try:
                if False: #sys.version_info > (2,7,9):
                    gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
                    data = urllib.urlopen('https://www.quantiacs.com/data/' +
                                          marketList[j]+'.txt',
                                          context=gcontext).read()
                else:
                    data = urllib.urlopen('https://www.quantiacs.com/data/' +
                                          marketList[j]+'.txt').read()

                with open(path, 'w') as dataFile:
                    dataFile.write(data)
                print 'Downloading ' + marketList[j]

            except:
                print 'Unable to download ' + marketList[j]
                marketList.remove(marketList[j])

    print 'Loading Data...'
    sys.stdout.flush()
    dataDict = {}
    largeDateRange = range(datetime.datetime(1990, 1, 1).toordinal(),
                           datetime.datetime.today().toordinal() + 1)
    DATE_Large = [int(datetime.datetime.fromordinal(j).strftime('%Y%m%d')) for j in largeDateRange]


    # Loading all markets into memory.
    for index, market in enumerate(marketList):
        marketFile = os.path.join('tickerData',market+'.txt')
        data = pd.read_csv(marketFile,engine='c')
        data.columns = map(str.strip,data.columns)
        data.set_index('DATE', inplace = True)
        data['DATE'] = data.index

        for index, dataType in enumerate(dataToLoad):
            if dataType == 'p':
                data.rename(columns={'p':'P'},inplace = True)
                dataType = 'P'

            if dataType != 'DATE' and not dataType in dataDict and dataType in data:
                dataDict[dataType] = pd.DataFrame(index=DATE_Large,columns=marketList)
                dataDict[dataType][market] = data[dataType]

            elif dataType != 'DATE' and dataType in data:
                dataDict[dataType][market] = data[dataType]

    dataDict['CLOSE'].dropna(how='all', inplace=True)

    # In-sample date management.
    if beginInSample is not None:
        beginInSample = datetime.datetime.strptime(beginInSample, '%Y%m%d')
    else:
        beginInSample = datetime.datetime(1990, 1, 1)
    beginInSampleInt = int(beginInSample.strftime('%Y%m%d'))

    if endInSample is not None:
        endInSample = datetime.datetime.strptime(endInSample, '%Y%m%d')
        endInSampleInt = int(endInSample.strftime('%Y%m%d'))
        dataDict['DATE'] = dataDict['CLOSE'].loc[beginInSampleInt:endInSampleInt, :].index.values
    else:
        dataDict['DATE'] = dataDict['CLOSE'].loc[beginInSampleInt:, :].index.values



    for index, dataType in enumerate(dataToLoad):
        if dataType != 'DATE' and dataType in dataDict:
            dataDict[dataType] = dataDict[dataType].loc[dataDict['DATE'], :]
            dataDict[dataType] = dataDict[dataType].values

    if 'VOL' in dataDict:
        dataDict['VOL'][np.isnan(dataDict['VOL'].astype(float))] = 0.0
    if 'OI' in dataDict:
        dataDict['OI'][np.isnan(dataDict['OI'].astype(float))] = 0.0
    if 'R' in dataDict:
        dataDict['R'][np.isnan(dataDict['R'].astype(float))]=0.0
    if 'RINFO' in dataDict:
        dataDict['RINFO'][np.isnan(dataDict['RINFO'].astype(float))]=0.0
        dataDict['RINFO']=dataDict['RINFO'].astype(float)
    if 'P' in dataDict:
        dataDict['P'][np.isnan(dataDict['P'].astype(float))]=0.0

    dataDict['CLOSE']=fillnans(dataDict['CLOSE'])

    dataDict['OPEN'], dataDict['HIGH'], dataDict['LOW'] = fillwith(dataDict['OPEN'],dataDict['CLOSE']), fillwith(dataDict['HIGH'],dataDict['CLOSE']), fillwith(dataDict['LOW'],dataDict['CLOSE'])

    print '\bDone! \n',
    sys.stdout.flush()

    return dataDict


def runts(tradingSystem, plotEquity=True, reloadData=False, state={}, sourceData='tickerData'):
    ''' backtests a trading system.

    evaluates the trading system function specified in the argument tsName and returns the struct ret. runts calls the trading system for each period with sufficient market data, and collets the returns of each call to compose a backtest.

    Example:

    # Might want to change this comment
    s = runts('tsName') evaluates the trading system specified in string tsName, and stores the result in struct s.

    Args:

        tsName (str): Specifies the trading system to be backtested
        plotEquity (bool, optional): Show the equity curve plot after the evaluation
        reloadData (bool,optional): Force reload of market data.
        state (dict, optional):  State information to resume computation of an existing backtest (for live evaluation on Quantiacs servers). State needs to be of the same form as ret.

    Returns:
        a dict mapping keys to the relevant backesting information: trading system name, system equity, trading dates, market exposure, market equity, the errorlog, the run time, the system's statistics, and the evaluation date.

        keys and description:
            'tsName' (str):    Name of the trading system, same as tsName
            'fundDate' (int):  All dates of the backtest in the format YYYYMMDD
            'fundEquity' (float):    Equity curve for the fund (collection of all markets)
            'returns' (float): Marketwise returns of trading system
            'marketEquity' (float):    Equity curves for each market in the fund
            'marketExposure' (float):    Collection of the returns p of the trading system function. Equivalent to the percent expsoure of each market in the fund. Normalized between -1 and 1
            'settings' (dict):    The settings of the trading system as defined in file tsName
            'errorLog' (list): list of strings with error messages
            'runtime' (float):    Runtime of the evaluation in seconds
            'stats' (dict): Performance numbers of the backtest
            'evalDate' (datetime): Last market data present in the backtest

    Copyright Quantiacs LLC - March 2015
    '''

    errorlog=[]
    ret={}

    if type(tradingSystem) is str:
        tradingSystem = tradingSystem.replace('\\', '/')

    filePathFlag = False
    if str(type(tradingSystem)) =="<type 'classobj'>" or str(type(tradingSystem)) =="<type 'type'>":
        TSobject = tradingSystem()
        settings = TSobject.mySettings()
        tsName = str(tradingSystem)

    elif str(type(tradingSystem)) == "<type 'instance'>" or str(type(tradingSystem)) == "<type 'module'>":
        TSobject = tradingSystem
        settings = TSobject.mySettings()
        tsName = str(tradingSystem)

    elif os.path.isfile(tradingSystem):
        filePathFlag = True
        filePath = str(tradingSystem)
        tsFolder, tsName = os.path.split(filePath)

        try:
            TSobject = imp.load_source('tradingSystemModule', filePath)
        except Exception as e:
            print 'Error loading trading system'
            print str(e)
            print traceback.format_exc()
            return

        try:
            settings = TSobject.mySettings()
        except Exception as e:
            print "Unable to load settings. Please ensure your settings definition is correct"
            print str(e)
            print traceback.format_exc()
            return

    else:
        print "Please input your trading system's file path or a callable object."
        return

    if isinstance(state, dict):
        if 'save' not in state:
            state['save']=False
        if 'resume' not in state:
            state['resume']=False
        if 'runtimeInterrupt' not in state:
            state['runtimeInterrupt'] = False
    else:
        print 'state variable is not a dict'

    # get boolean index of futures
    futuresIx=np.array(map(lambda string:bool(re.match("F_",string)),settings['markets']))

    # get data fields and extract them.
    requiredData = set(['DATE','OPEN','HIGH', 'LOW', 'CLOSE', 'P','RINFO','p'])
    dataToLoad = requiredData

    tsArgs = inspect.getargspec(TSobject.myTradingSystem)
    tsArgs = tsArgs[0]
    tsDataToLoad = [item for index, item in enumerate(tsArgs) if item.isupper()]

    dataToLoad.update(tsDataToLoad)

    global settingsCache
    global dataCache

    if 'settingsCache' not in globals() or settingsCache != settings:
        if 'beginInSample' in settings and 'endInSample' in settings:
            dataDict=loadData(settings['markets'],dataToLoad,reloadData, beginInSample = settings['beginInSample'], endInSample = settings['endInSample'], dataDir=sourceData)
        elif 'beginInSample' in settings and 'endInSample' not in settings:
            dataDict=loadData(settings['markets'],dataToLoad,reloadData, settings['beginInSample'], dataDir=sourceData)
        elif 'endInSample' in settings and 'beginInSample' not in settings:
            dataDict=loadData(settings['markets'],dataToLoad,reloadData, endInSample = settings['endInSample'], dataDir=sourceData)
        else:
            dataDict=loadData(settings['markets'],dataToLoad,reloadData, dataDir=sourceData)

        dataCache=deepcopy(dataDict)
        settingsCache = deepcopy(settings)

    else:
        print 'copying data from cache'
        settings= deepcopy(settingsCache)
        dataDict = deepcopy(dataCache)

    print 'Evaluating Trading System'

    nMarkets=len(settings['markets'])
    endLoop=len(dataDict['DATE'])

    if 'RINFO' in dataDict:
        Rix= dataDict['RINFO'] != 0
    else:
        dataDict['RINFO'] = np.zeros(np.shape(dataDict['CLOSE']))
        Rix = np.zeros(np.shape(dataDict['CLOSE']))



    dataDict['exposure']=np.zeros((endLoop,nMarkets))
    dataDict['equity']=np.ones((endLoop,nMarkets))
    dataDict['fundEquity'] = np.ones((endLoop,1))
    realizedP = np.zeros((endLoop, nMarkets))
    returns = np.zeros((endLoop, nMarkets))

    sessionReturnTemp = np.append( np.empty((1,nMarkets))*np.nan,(( dataDict['CLOSE'][1:,:]- dataDict['OPEN'][1:,:]) / dataDict['CLOSE'][0:-1,:] ), axis =0 ).copy()
    sessionReturn=np.nan_to_num( fillnans(sessionReturnTemp) )
    gapsTemp=np.append(np.empty((1,nMarkets))*np.nan, (dataDict['OPEN'][1:,:]- dataDict['CLOSE'][:-1,:]-dataDict['RINFO'][1:,:].astype(float)) / dataDict['CLOSE'][:-1:],axis=0)
    gaps=np.nan_to_num(fillnans(gapsTemp))

    slippageTemp = np.append(np.empty((1,nMarkets))*np.nan, ((dataDict['HIGH'][1:,:] - dataDict['LOW'][1:,:]) / dataDict['CLOSE'][:-1,:] ), axis=0) * settings['slippage']
    SLIPPAGE = np.nan_to_num(fillnans(slippageTemp))
#    gaps[Rix] = 0

    if 'lookback' not in settings:
        startLoop=2
        settings['lookback']=1
    else:
        startLoop= settings['lookback']-1

    # Server evaluation --- resumes for new day.
    if state['resume']:
        if 'evalData' in state:
            ixOld= dataDict['DATE']<=state['evalData']['evalDate']
            evalData=state['evalData']

            ixMapExposure=np.concatenate(([False,False],ixOld),axis=0)
            dataDict['equity'][ixOld,:]=state['evalData']['marketEquity']
            dataDict['exposure'][ixMapExposure,:]=state['evalData']['marketExposure']
            dataDict['fundEquity'][ixOld,:] = state['evalData']['fundEquity']

            startLoop = np.shape(state['evalData']['fundDate'])[0]
            endLoop = np.shape(dataDict['DATE'])[0]

            print('Resuming'+tsName+' | computing '+str(endLoop-startLoop+1)+' new days')
            settings= evalData['settings']

    t0= time.time()

    # Loop through trading days
    for t in range(startLoop,endLoop):
        todaysP= dataDict['exposure'][t-1,:]
        yesterdaysP = realizedP[t-2,:]
        deltaP=todaysP-yesterdaysP

        newGap=yesterdaysP * gaps[t,:]
        newGap[np.isnan(newGap)]= 0

        newRet = todaysP * sessionReturn[t,:] - abs(deltaP * SLIPPAGE[t,:])
        newRet[np.isnan(newRet)] = 0

        returns[t,:] = newRet + newGap
        dataDict['equity'][t,:] = dataDict['equity'][t-1,:] * (1+returns[t,:])
        dataDict['fundEquity'][t] = (dataDict['fundEquity'][t-1] * (1+np.sum(returns[t,:])))

        realizedP[t-1,:] = dataDict['CLOSE'][t,:] / dataDict['CLOSE'][t-1,:] * dataDict['fundEquity'][t-1] / dataDict['fundEquity'][t] * todaysP

        # Roll futures contracts.
        if np.any(Rix[t,:]):
            delta=np.tile(dataDict['RINFO'][t,Rix[t,:]],(t,1))
            dataDict['CLOSE'][0:t,Rix[t,:]] = dataDict['CLOSE'][0:t,Rix[t,:]].copy() + delta.copy()
            dataDict['OPEN'][0:t,Rix[t,:]] = dataDict['OPEN'][0:t,Rix[t,:]].copy()  + delta.copy()
            dataDict['HIGH'][0:t,Rix[t,:]] = dataDict['HIGH'][0:t,Rix[t,:]].copy() + delta.copy()
            dataDict['LOW'][0:t,Rix[t,:]] = dataDict['LOW'][0:t,Rix[t,:]].copy()   + delta.copy()

        try:
            argList= []

            for index in range(len(tsArgs)):
                if tsArgs[index]=='settings':
                    argList.append(settings)
                elif tsArgs[index] == 'self':
                    continue
                else:
                    argList.append(dataDict[tsArgs[index]][t- settings['lookback'] +1:t+1].copy())

            position, settings= TSobject.myTradingSystem(*argList)
        except:
            print 'Error evaluating trading system'
            print sys.exc_info()[0]
            print traceback.format_exc()
            errorlog.append(str(dataDict['DATE'][t])+ ': ' + str(sys.exc_info()[0]))
            dataDict['equity'][t:,:] = np.tile(dataDict['equity'][t,:],(endLoop-t,1))
            return
        position[np.isnan(position)] = 0
        position = np.real(position)
        position = position/np.sum(abs(position))
        position[np.isnan(position)] = 0  # extra nan check in case the positions sum to zero

        dataDict['exposure'][t,:] = position.copy()

        t1=time.time()
        runtime = t1-t0
        if runtime > 300 and state['runtimeInterrupt']:
            errorlog.append('Evaluation stopped: Runtime exceeds 5 minutes.')
            break

    if 'budget' in settings:
        fundequity = dataDict['fundEquity'][(settings['lookback']-1):,:] * settings['budget']
    else:
        fundequity = dataDict['fundEquity'][(settings['lookback']-1):,:]

    marketRets = np.float64(dataDict['CLOSE'][1:,:] - dataDict['CLOSE'][:-1,:] - dataDict['RINFO'][1:,:])/dataDict['CLOSE'][:-1,:]
    marketRets = fillnans(marketRets)
    marketRets[np.isnan(marketRets)] = 0
    marketRets = marketRets.tolist()
    a = np.zeros((1,nMarkets))
    a = a.tolist()
    marketRets = a + marketRets

    ret['returns'] = np.nan_to_num(returns).tolist()

    if errorlog:
	    print 'Error: {}'.format(errorlog)

    if plotEquity:
        statistics = stats(fundequity)

        returns = plotts(tradingSystem, fundequity,dataDict['equity'], dataDict['exposure'], settings, dataDict['DATE'][settings['lookback']-1:], statistics,ret['returns'],marketRets)

    else:
        statistics= stats(fundequity)


    ret['tsName']=tsName
    ret['fundDate']=dataDict['DATE'].tolist()
    ret['fundEquity']=dataDict['fundEquity'].tolist()
    ret['marketEquity']= dataDict['equity'].tolist()
    ret['marketExposure'] = dataDict['exposure'].tolist()
    ret['errorLog']=errorlog
    ret['runtime']=runtime
    ret['stats']=statistics
    ret['settings']=settings
    ret['evalDate']=dataDict['DATE'][t]

    if state['save']:
        with open(tsName+'.json', 'w+') as fileID:
            stateSave=json.dump(ret,fileID)
    return ret


def plotts(tradingSystem, equity,mEquity,exposure,settings,DATE,statistics,returns,marketReturns):
    ''' plots equity curve and calculates trading system statistics

    Args:
        equity (list): list of equity of evaluated trading system.
        mEquity (list): list of equity of each market over the trading days.
        exposure (list): list of positions over the trading days.
        settings (dict): list of settings.
        DATE (list): list of dates corresponding to entries in equity.

    Copyright Quantiacs LLC - March 2015
    '''
    # Initialize selected index of the two dropdown lists
    style.use("ggplot")
    global indx_TradingPerf, indx_Exposure, indx_MarketRet
    inx = [0]
    inx2 = [0]
    inx3 = [0]
    indx_TradingPerf = 0
    indx_Exposure = 0
    indx_MarketRet = 0
    mRetMarkets = settings['markets'][1:]
    settings['markets'].insert(0,'fundEquity')

    DATEord=[]
    lng = len(DATE)
    for i in range(lng):
        DATEord.append(datetime.datetime.strptime(str(DATE[i]),'%Y%m%d'))

    # Prepare all the y-axes
    equityList = np.transpose(np.array(mEquity))

    Long = np.transpose(np.array(exposure))
    Long[Long<0] = 0
    Long = Long[:,(settings['lookback']-2):-1]     # Market Exposure lagged by one day

    Short = - np.transpose(np.array(exposure))
    Short[Short<0] = 0
    Short = Short[:,(settings['lookback']-2):-1]


    returnsList = np.transpose(np.array(returns))

    returnLong = np.transpose(np.array(exposure))
    returnLong[returnLong<0] = 0
    returnLong[returnLong > 0] = 1
    returnLong = np.multiply(returnLong[:,(settings['lookback']-2):-1],returnsList[:,(settings['lookback']-1):])      # y values for Long Only Equity Curve


    returnShort = - np.transpose(np.array(exposure))
    returnShort[returnShort<0] = 0
    returnShort[returnShort > 0] = 1
    returnShort = np.multiply(returnShort[:,(settings['lookback']-2):-1],returnsList[:,(settings['lookback']-1):])        # y values for Short Only Equity Curve

    marketRet = np.transpose(np.array(marketReturns))
    marketRet = marketRet[:,(settings['lookback']-1):]
    equityList = equityList[:,(settings['lookback']-1):]    # y values for all individual markets



    def plot(indx_TradingPerf,indx_Exposure):
        plt.clf()

        Subplot_Equity = plt.subplot2grid((8,8), (0,0), colspan = 6, rowspan = 6)
        Subplot_Exposure = plt.subplot2grid((8,8), (6,0), colspan = 6, rowspan = 2, sharex = Subplot_Equity)
        t = np.array(DATEord)

        if indx_TradingPerf == 0:   # fundEquity selected
            lon = Long.sum(axis=0)
            sho = Short.sum(axis=0)
            y_Long = lon
            y_Short = sho
            if indx_Exposure == 0:  # Long & Short selected
                y_Equity = equity
                Subplot_Equity.plot(t,y_Equity,'b',linewidth=0.5)
            elif indx_Exposure == 1:  # Long Selected
                y_Equity = settings['budget'] * np.cumprod(1+returnLong.sum(axis = 0))
                Subplot_Equity.plot(t,y_Equity,'c',linewidth=0.5)
            else:      # Short Selected
                y_Equity = settings['budget'] * np.cumprod(1+returnShort.sum(axis = 0))
                Subplot_Equity.plot(t,y_Equity,'g',linewidth=0.5)
            statistics=stats(y_Equity)
            Subplot_Equity.plot(DATEord[statistics['maxDDBegin']:statistics['maxDDEnd']+1],y_Equity[statistics['maxDDBegin']:statistics['maxDDEnd']+1],color='red',linewidth=0.5, label = 'Max Drawdown')
            # Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
            if not(np.isnan(statistics['maxTimeOffPeakBegin'])) and not(np.isnan(statistics['maxTimeOffPeak'])):
                Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
            Subplot_Exposure.plot(t,y_Long,'c',linewidth=0.5, label = 'Long')
            Subplot_Exposure.plot(t,y_Short,'g',linewidth=0.5, label = 'Short')
            # Hide the Long(Short) curve in market exposure subplot when Short(Long) is plotted in Equity Curve subplot
            if indx_Exposure == 1:
                Subplot_Exposure.lines.pop(1)
            elif indx_Exposure == 2:
                Subplot_Exposure.lines.pop(0)
            Subplot_Equity.set_yscale('log')
            Subplot_Equity.set_ylabel('Performance (Logarithmic)')
        else:   # individual market selected
            y_Long = Long[indx_TradingPerf-1]
            y_Short = Short[indx_TradingPerf-1]
            if indx_Exposure == 0:            # Long & Short Selected
                y_Equity = equityList[indx_TradingPerf-1]
                Subplot_Equity.plot(t,y_Equity,'b',linewidth=0.5)
            elif indx_Exposure == 1:        # Long Selected
                y_Equity = np.cumprod(1+returnLong[indx_TradingPerf-1])
                Subplot_Equity.plot(t,y_Equity,'c',linewidth=0.5)
            else:                   # Short Selected
                y_Equity = np.cumprod(1+returnShort[indx_TradingPerf-1])
                Subplot_Equity.plot(t,y_Equity,'g',linewidth=0.5)
            statistics=stats(y_Equity)
            Subplot_Exposure.plot(t,y_Long,'c',linewidth=0.5, label = 'Long')
            Subplot_Exposure.plot(t,y_Short,'g',linewidth=0.5, label = 'Short')
            if indx_Exposure == 1:
                Subplot_Exposure.lines.pop(1)
            elif indx_Exposure == 2:
                Subplot_Exposure.lines.pop(0)
            if np.isnan(statistics['maxDDBegin']) == False:
                Subplot_Equity.plot(DATEord[statistics['maxDDBegin']:statistics['maxDDEnd']+1],y_Equity[statistics['maxDDBegin']:statistics['maxDDEnd']+1],color='red',linewidth=0.5, label = 'Max Drawdown')
                if not(np.isnan(statistics['maxTimeOffPeakBegin'])) and not(np.isnan(statistics['maxTimeOffPeak'])):
                    Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
                # Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
            Subplot_Equity.set_ylabel('Performance')

        statsStr="Sharpe Ratio = {sharpe:.4f}\nSortino Ratio = {sortino:.4f}\n\nPerformance (%/yr) = {returnYearly:.4f}\nVolatility (%/yr)       = {volaYearly:.4f}\n\nMax Drawdown = {maxDD:.4f}\nMAR Ratio         = {mar:.4f}\n\n Max Time off peak =  {maxTimeOffPeak}\n\n\n\n\n\n".format(**statistics)


        Subplot_Equity.autoscale(tight=True)
        Subplot_Exposure.autoscale(tight=True)
        Subplot_Equity.set_title('Trading Performance of %s' %settings['markets'][indx_TradingPerf])
        Subplot_Equity.get_xaxis().set_visible(False)
        Subplot_Exposure.set_ylabel('Long/Short')
        Subplot_Exposure.set_xlabel('Year')
        Subplot_Equity.legend(bbox_to_anchor=(1.03, 0), loc='lower left', borderaxespad=0.)
        Subplot_Exposure.legend(bbox_to_anchor=(1.03, 0.63), loc='lower left', borderaxespad=0.)

        # Performance Numbers Textbox
        f.text(.72,.58,statsStr)

        plt.gcf().canvas.draw()


    def plot2(indx_Exposure, indx_MarketRet):
        plt.clf()

        MarketReturns = plt.subplot2grid((8,8), (0,0), colspan = 6, rowspan = 8)
        t = np.array(DATEord)

        if indx_Exposure == 2:
            mRet = np.cumprod(1-marketRet[indx_MarketRet+1])
        else:
            mRet = np.cumprod(1+marketRet[indx_MarketRet+1])

        MarketReturns.plot(t,mRet,'b',linewidth=0.5)
        statistics=stats(mRet)
        MarketReturns.set_ylabel('Market Returns')

        statsStr="Sharpe Ratio = {sharpe:.4f}\nSortino Ratio = {sortino:.4f}\n\nPerformance (%/yr) = {returnYearly:.4f}\nVolatility (%/yr)       = {volaYearly:.4f}\n\nMax Drawdown = {maxDD:.4f}\nMAR Ratio         = {mar:.4f}\n\n Max Time off peak =  {maxTimeOffPeak}\n\n\n\n\n\n".format(**statistics)

        MarketReturns.autoscale(tight=True)
        MarketReturns.set_title('Market Returns of %s' %mRetMarkets[indx_MarketRet])
        MarketReturns.set_xlabel('Date')

        # Performance Numbers Textbox
        f.text(.72,.58,statsStr)

        plt.gcf().canvas.draw()



    # Callback function for two dropdown lists
    def newselection(event):
        global indx_TradingPerf, indx_Exposure, indx_MarketRet
        value_of_combo = dropdown.current()
        inx.append(value_of_combo)
        indx_TradingPerf = inx[-1]
        indx_MarketRet = -1

        plot(indx_TradingPerf,indx_Exposure)

    def newselection2(event):
        global indx_TradingPerf, indx_Exposure, indx_MarketRet
        value_of_combo2 = dropdown2.current()
        inx2.append(value_of_combo2)
        indx_Exposure = inx2[-1]

        if indx_TradingPerf == -1:
            plot2(indx_Exposure,indx_MarketRet)
        else:
            plot(indx_TradingPerf,indx_Exposure)

    def newselection3(event):
        global indx_TradingPerf, indx_Exposure, indx_MarketRet
        value_of_combo3 = dropdown3.current()
        inx3.append(value_of_combo3)
        indx_MarketRet = inx3[-1]
        indx_TradingPerf = -1

        plot2(indx_Exposure, indx_MarketRet)

    def submit_callback():
        tsfolder, tsName = os.path.split(tradingSystem)
        submit(tradingSystem,tsName[:-3])


    def shutdown_interface():
        TradingUI.eval('::ttk::CancelRepeat')
        # TradingUI.destroy()
        TradingUI.quit()
        TradingUI.destroy()
        # sys.exit()

    # GUI mainloop
    TradingUI = tk.Tk()
    TradingUI.title('Trading System Performance')

    Label_1 = tk.Label(TradingUI, text="Trading Performance:")
    Label_1.grid(row = 0, column = 0, sticky = tk.EW)

    box_value = tk.StringVar()
    dropdown = ttk.Combobox(TradingUI, textvariable  = box_value, state = 'readonly')
    dropdown['values'] = settings['markets']
    dropdown.grid(row=0, column=1,sticky=tk.EW)
    dropdown.current(0)
    dropdown.bind('<<ComboboxSelected>>',newselection)

    Label_2 = tk.Label(TradingUI, text="Exposure:")
    Label_2.grid(row = 0, column = 2, sticky = tk.EW)

    box_value2 = tk.StringVar()
    dropdown2 = ttk.Combobox(TradingUI, textvariable  = box_value2, state = 'readonly')
    dropdown2['values'] = ['Long & Short', 'Long', 'Short']
    dropdown2.grid(row=0, column=3,sticky=tk.EW)
    dropdown2.current(0)
    dropdown2.bind('<<ComboboxSelected>>',newselection2)

    Label_3 = tk.Label(TradingUI, text="Market Returns:")
    Label_3.grid(row = 0, column = 4, sticky = tk.EW)

    box_value3 = tk.StringVar()
    dropdown3 = ttk.Combobox(TradingUI, textvariable  = box_value3, state = 'readonly')
    dropdown3['values'] = mRetMarkets
    dropdown3.grid(row=0, column=5,sticky=tk.EW)
    dropdown3.current(0)
    dropdown3.bind('<<ComboboxSelected>>',newselection3)

    f = plt.figure(figsize = (14,8))
    canvas = FigureCanvasTkAgg(f, master=TradingUI)



    if updateCheck():
        Text1 = tk.Entry(TradingUI)
        Text1.insert(0, "Toolbox update available. Run 'pip install --upgrade quantiacsToolbox' from the command line to upgrade.")
        Text1.configure(justify = 'center', state = 'readonly')

        Text1.grid(row  = 1, column = 0, columnspan = 6, sticky = tk.EW)
        canvas.get_tk_widget().grid(row=2,column=0,columnspan = 6,sticky=tk.NSEW)

    else:
        canvas.get_tk_widget().grid(row=1,column=0,columnspan = 6, rowspan = 2, sticky=tk.NSEW)


    button_submit = tk.Button(TradingUI, text = 'Submit Trading System', command = submit_callback)
    button_submit.grid(row = 4, column = 0, columnspan = 6, sticky = tk.EW)

    Subplot_Equity = plt.subplot2grid((8,8), (0,0), colspan = 6, rowspan = 6)
    Subplot_Exposure = plt.subplot2grid((8,8), (6,0), colspan = 6, rowspan = 2, sharex = Subplot_Equity)
    t = np.array(DATEord)

    lon = Long.sum(axis=0)
    sho = Short.sum(axis=0)
    y_Long = lon
    y_Short = sho
    y_Equity = equity
    Subplot_Equity.plot(t,y_Equity,'b',linewidth=0.5)
    statistics=stats(y_Equity)
    Subplot_Equity.plot(DATEord[statistics['maxDDBegin']:statistics['maxDDEnd']+1],y_Equity[statistics['maxDDBegin']:statistics['maxDDEnd']+1],color='red',linewidth=0.5, label = 'Max Drawdown')
    # Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
    if not(np.isnan(statistics['maxTimeOffPeakBegin'])) and not(np.isnan(statistics['maxTimeOffPeak'])):
        Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
    Subplot_Exposure.plot(t,y_Long,'c',linewidth=0.5, label = 'Long')
    Subplot_Exposure.plot(t,y_Short,'g',linewidth=0.5, label = 'Short')
    Subplot_Equity.set_yscale('log')
    Subplot_Equity.set_ylabel('Performance (Logarithmic)')

    statsStr="Sharpe Ratio = {sharpe:.4f}\nSortino Ratio = {sortino:.4f}\n\nPerformance (%/yr) = {returnYearly:.4f}\nVolatility (%/yr)       = {volaYearly:.4f}\n\nMax Drawdown = {maxDD:.4f}\nMAR Ratio         = {mar:.4f}\n\n Max Time off peak =  {maxTimeOffPeak}\n\n\n\n\n\n".format(**statistics)

    Subplot_Equity.autoscale(tight=True)
    Subplot_Exposure.autoscale(tight=True)
    Subplot_Equity.set_title('Trading Performance of %s' %settings['markets'][indx_TradingPerf])
    Subplot_Equity.get_xaxis().set_visible(False)
    Subplot_Exposure.set_ylabel('Long/Short')
    Subplot_Exposure.set_xlabel('Year')
    Subplot_Equity.legend(bbox_to_anchor=(1.03, 0), loc='lower left', borderaxespad=0.)
    Subplot_Exposure.legend(bbox_to_anchor=(1.03, 0.63), loc='lower left', borderaxespad=0.)

    plt.gcf().canvas.draw()
    f.text(.72,.58,statsStr)


    TradingUI.protocol("WM_DELETE_WINDOW", shutdown_interface)

    TradingUI.mainloop()




def stats(equityCurve):
    ''' calculates trading system statistics

    Calculates and returns a dict containing the following statistics
    - sharpe ratio
    - sortino ratio
    - annualized returns
    - annualized volatility
    - maximum drawdown
        - the dates at which the drawdown begins and ends
    - the MAR ratio
    - the maximum time below the peak value
        - the dates at which the max time off peak begin and end

    Args:
        equityCurve (list): the equity curve of the evaluated trading system

    Returns:
        statistics (dict): a dict mapping keys to corresponding trading system statistics (sharpe ratio, sortino ration, max drawdown...)

    Copyright Quantiacs LLC - March 2015

    '''
    returns = (equityCurve[1:]-equityCurve[:-1])/equityCurve[:-1]

    volaDaily=np.std(returns)
    volaYearly=np.sqrt(252)*volaDaily

    index=np.cumprod(1+returns)
    indexEnd=index[-1]

    returnDaily = np.exp(np.log(indexEnd)/returns.shape[0])-1
    returnYearly = (1+returnDaily)**252-1
    sharpeRatio = returnYearly / volaYearly

    downsideReturns = returns.copy()
    downsideReturns[downsideReturns > 0]= 0
    downsideVola = np.std(downsideReturns)
    downsideVolaYearly = downsideVola *np.sqrt(252)

    sortino = returnYearly / downsideVolaYearly

    highCurve = equityCurve.copy()

    testarray = np.ones((1,len(highCurve)))
    test = np.array_equal(highCurve,testarray[0])

    if test:
        mX = np.NaN
        mIx = np.NaN
        maxDD = np.NaN
        mar = np.NaN
        maxTimeOffPeak = np.NaN
        mtopStart = np.NaN
        mtopEnd = np.NaN
    else:
        for k in range(len(highCurve)-1):
            if highCurve[k+1] < highCurve[k]:
                highCurve[k+1] = highCurve[k]

        underwater = equityCurve / highCurve
        mi = np.min(underwater)
        mIx = np.argmin(underwater)
        maxDD = 1 - mi
        mX= np.where(highCurve[0:mIx-1] == np.max(highCurve[0:mIx-1]))
#        highList = highCurve.copy()
#        highList.tolist()
#        mX= highList[0:mIx].index(np.max(highList[0:mIx]))
        mX=mX[0][0]
        mar   = returnYearly / maxDD

        mToP = equityCurve < highCurve
        mToP = np.insert(mToP, [0,len(mToP)],False)
        mToPdiff=np.diff(mToP.astype('int'))
        ixStart   = np.where(mToPdiff==1)[0]
        ixEnd     = np.where(mToPdiff==-1)[0]

        offPeak         = ixEnd - ixStart
        if len(offPeak) > 0:
            maxTimeOffPeak  = np.max(offPeak)
            topIx           = np.argmax(offPeak)
        else:
            maxTimeOffPeak = 0
            topIx          = np.zeros(0)

        if np.not_equal(np.size(topIx),0):
            mtopStart= ixStart[topIx]-2
            mtopEnd= ixEnd[topIx]-1

        else:
            mtopStart = np.NaN
            mtopEnd = np.NaN
            maxTimeOffPeak = np.NaN

    statistics={}
    statistics['sharpe']              = sharpeRatio
    statistics['sortino']             = sortino
    statistics['returnYearly']        = returnYearly
    statistics['volaYearly']          = volaYearly
    statistics['maxDD']               = maxDD
    statistics['maxDDBegin']          = mX
    statistics['maxDDEnd']            = mIx
    statistics['mar']                 = mar
    statistics['maxTimeOffPeak']      = maxTimeOffPeak
    statistics['maxTimeOffPeakBegin'] = mtopStart
    statistics['maxTimeOffPeakEnd']   = mtopEnd

    return statistics


def submit(tradingSystem, tsName):
    ''' submits trading system to Quantiacs server

    Args:
        tradingSystem (file, obj, instance): accepts a filepath, a class object, or class instance.
        tsName (str): the desired trading system name for display on Quantiacs website.

    Returns:
        returns True if upload was successful, False otherwise.

    '''
    from version import __version__


    if os.path.isfile(tradingSystem) and os.access(tradingSystem, os.R_OK):
        filePathFlag=True
        filePath=tradingSystem
        fileFolder, fileName=os.path.split(filePath)

    else:
        print "Please input the your trading system's file path."

    toolboxPath=os.path.realpath(__file__)
    toolboxDir,Nothing=os.path.split(toolboxPath)

    print "Submitting File..."
    fid=open(filePath)
    fileText=fid.read()
    fid.close()

    if sys.version_info > (2,7,9):
        uploadContext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
        submissionUrl='https://www.quantiacs.com/quantnetsite/UploadTradingSystem.aspx'
        data=urllib.urlencode({'fileName':fileName[:-3],'name':tsName,'data':fileText, 'version':__version__})
        req = urllib2.Request(submissionUrl, data)
        guid= urllib2.urlopen(req, context = uploadContext)
    else:
        submissionUrl='http://www.quantiacs.com/quantnetsite/UploadTradingSystem.aspx'
        data=urllib.urlencode({'fileName':fileName[:-3],'name':tsName,'data':fileText, 'version':__version__})
        req = urllib2.Request(submissionUrl, data)
        guid= urllib2.urlopen(req,)


    successPage = guid.read()

    webbrowser.open_new_tab('https://www.quantiacs.com/quantnetsite/UploadSuccess.aspx?guid='+str(successPage))

#    return True





def computeFees(equityCurve, managementFee,performanceFee):
    ''' computes equity curve after fees

    Args:
        equityCurve (list, numpy array) : a column vector of daily fund values
        managementFee (float) : the management fee charged to the investor (a portion of the AUM charged yearly)
        performanceFee (float) : the performance fee charged to the investor (the portion of the difference between a new high and the most recent high, charged daily)

    Returns:
        returns an equity curve with the fees subtracted.  (does not include the effect of fees on equity lot size)

    '''
    returns = (np.array(equityCurve[1:])-np.array(equityCurve[:-1]))/np.array(equityCurve[:-1])
    ret = np.append(0,returns)

    tradeDays = ret > 0
    firstTradeDayRow = np.where(tradeDays is True)
    firstTradeDay = firstTradeDayRow[0][0]

    manFeeIx = np.zeros(np.shape(ret),dtype=bool)
    manFeeIx[firstTradeDay:] = 1
    ret[manFeeIx] = ret[manFeeIx] - managementFee/252

    ret = 1 + ret
    r = np.ndarray((0,0))
    high = 1
    last = 1
    pFee = np.zeros(np.shape(ret))
    mFee = np.zeros(np.shape(ret))

    for k in range(len(ret)):
        mFee[k] = last * managementFee/252 * equityCurve[0][0]
        if last * ret[k] > high:
            iFix = high / last
            iPerf = ret[k] / iFix
            pFee[k] = (iPerf - 1) * performanceFee * iFix * equityCurve[0][0]
            iPerf = 1 + (iPerf - 1) * (1-performanceFee)
            r=np.append(r,iPerf * iFix)
        else:
            r=np.append(r,ret[k])
        if np.size(r)>0:
            last = r[-1] * last
        if last > high:
            high = last

    out = np.cumprod(r)
    out = out * equityCurve[0]

    return out


def fillnans(inArr):
    ''' fills in (column-wise)value gaps with the most recent non-nan value.

    fills in value gaps with the most recent non-nan value.
    Leading nan's remain in place. The gaps are filled in only after the first non-nan entry.

    Args:
        inArr (list, numpy array)
    Returns:
        returns an array of the same size as inArr with the nan-values replaced by the most recent non-nan entry.

    '''
    inArr=inArr.astype(float)
    nanPos= np.where(np.isnan(inArr))
    nanRow=nanPos[0]
    nanCol=nanPos[1]
    myArr=inArr.copy()
    for i in range(len(nanRow)):
        if nanRow[i] >0:
            myArr[nanRow[i],nanCol[i]]=myArr[nanRow[i]-1,nanCol[i]]
    return myArr


def fillwith(field, lookup):
    ''' replaces nan entries of field, with values of lookup.

    Args:
        field (list, numpy array) : array whose nan-values are to be replaced
        lookup (list, numpy array) : array to copy values for placement in field

    Returns:
        returns array with nan-values replaced by entries in lookup.
    '''

    out = field.astype(float)
    nanPos= np.where(np.isnan(out))
    nanRow=nanPos[0]
    nanCol=nanPos[1]

    for i in range(len(nanRow)):
        out[nanRow[i],nanCol[i]] = lookup[nanRow[i]-1,nanCol[i]]

    return out


def ismember(a, b):
    bIndex = {}
    for item, elt in enumerate(b):
        if elt not in bIndex:
            bIndex[elt] = item
    return [bIndex.get(item, None) for item in a]


def updateCheck():
    ''' checks for new version of toolbox

    Returns:
        returns True if the version of the toolox on PYPI is not the same as the current version
        returns False if version is the same
    '''

    from version import __version__
    updateStr = ''
    try:
        toolboxJson = urllib.urlopen('https://pypi.python.org/pypi/quantiacsToolbox/json')
    except Exception as e:
        return False

    toolboxDict = json.loads(toolboxJson.read())

    if __version__ != toolboxDict['info']['version']:
        return True
    else:
        return False
