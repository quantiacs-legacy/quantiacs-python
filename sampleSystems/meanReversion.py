### Quantiacs Mean Reversion Trading System Example
# import necessary Packages below:
import numpy

##### Do not change this function definition #####
def myTradingSystem(DATE, CLOSE, settings):
    ''' This system uses mean reversion techniques to allocate capital into the desired equities '''

    # This strategy evaluates two averages over time of the close over a long/short
    # scale and builds the ratio. For each day, "smaQuot" is an array of "nMarkets"
    # size.
    nMarkets = numpy.shape(CLOSE)[1]
    periodLong= 200
    periodShort= 40

    smaLong=   numpy.sum(CLOSE[-periodLong:,:], axis=0)/periodLong
    smaRecent= numpy.sum(CLOSE[-periodShort:,:],axis=0)/periodShort
    smaQuot= smaRecent / smaLong

    # For each day, scan the ratio of moving averages over the markets and find the
    # market with the maximum ratio and the market with the minimum ratio:
    longEquity = numpy.where(smaQuot == numpy.nanmin(smaQuot))
    shortEquity= numpy.where(smaQuot == numpy.nanmax(smaQuot))

    # Take a contrarian view, going long the market with the minimum ratio and
    # going short the market with the maximum ratio. The array "pos" will contain
    # all zero entries except for those cases where we go long (1) and short (-1):
    pos= numpy.zeros((1,nMarkets))
    pos[0,longEquity[0][0]] = 0
    pos[0,shortEquity[0][0]]= -1

    # For the position sizing, we supply a vector of weights defining our
    # exposure to the markets in settings['markets']. This vector should be
    # normalized.
    pos= pos/numpy.nansum(abs(pos))

    return pos, settings


##### Do not change this function definition #####
def mySettings():
    ''' Define your trading system settings here '''
    settings= {}

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
    settings['markets']  = ['CASH','F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC',
    'F_CD', 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV',
    'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD',
    'F_MP', 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB',
    'F_RU', 'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US',
    'F_W', 'F_XX', 'F_YM']


    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
