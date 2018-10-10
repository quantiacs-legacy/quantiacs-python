import numpy as np


def LAG(field, period):
    nMarkets = np.shape(field)[1]
    out =  np.append(np.zeros((period,nMarkets))*np.nan, field[:-period,:],axis=0)
    return out

def ATR(HIGH, LOW, CLOSE, period):

    tr = TR(HIGH,LOW,CLOSE)
    out = np.mean(tr[-period:,:],axis=0)
    return out

def TR(HIGH, LOW, CLOSE):
    CLOSELAG = LAG(CLOSE,1)
    range1 = HIGH - LOW
    range2 = np.abs(HIGH-CLOSELAG)
    range3 = np.abs(LOW -CLOSELAG)
    out = np.fmax(np.fmax(range1,range2),range3)
    return out

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, settings, exposure):
    if 'long' not in settings:
        settings['long']=0
        settings['short']=0

    nMarkets = len(settings['markets'])
    p = np.zeros(nMarkets)

    holding = 4

    if settings['short'] != 0:
        settings['short'] = (settings['short'] % holding) +1
        p[0],p[1] = 0,-1
    if settings['short'] == 0 or settings['short']==4:
        p[0],p[1] = 1,0
        settings['short'] = 0


    if settings['long'] != 0:
        settings['long'] = (settings['long'] % holding) +1
        p[0],p[1] = 0,1
    if settings['long'] == 0 or settings['long']==4:
        p[0],p[1] = 1,0
        settings['long'] = 0


    closeRange = np.ptp(CLOSE[-4:,1])
    atr9 = ATR(HIGH, LOW, CLOSE, 9)
    atr1 = ATR(HIGH, LOW, CLOSE, 1)

    LongRule1 = atr9[1] < atr1[1]
    LongRule2 = CLOSE[-1,1] <= min(CLOSE[-9:,1])
    LongRule3 = CLOSE[-1,2] > CLOSE[-9,2]

    ShortRule1 = atr9[1]< atr1[1]
    ShortRule2 = CLOSE[-1,1] >= max(CLOSE[-9:,1])
    ShortRule3 = CLOSE[-1,2] < CLOSE[-9,2]

    if LongRule1 and LongRule2 and LongRule3:
        p[0], p[1] = 0, 1
        settings['long'] = (settings['long'] % holding) +1
        settings['short'] = 0

    if ShortRule1 and ShortRule2 and ShortRule3:
        p[0], p[1] = 0, -1
        settings['short'] = (settings['short'] % holding) +1
        settings['long'] = 0
    return p, settings


def mySettings():
    settings = {}
    settings['markets']     = ['CASH','F_NG', 'F_CL']
    settings['slippage']    = 0.0
    settings['budget']      = 1000000
    settings['beginInSample'] = '20040101'
    settings['endInSample']   = '20200101'
    settings['lookback']    = 504

    return settings

if __name__=='__main__':
    from quantiacsToolbox import runts
    results = runts(__file__)
