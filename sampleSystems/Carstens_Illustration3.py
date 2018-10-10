import numpy as np
import datetime

def LAG(field, period):
    nMarkets = np.shape(field)[1]
    out =  np.append(np.zeros((period,nMarkets))*np.nan,field[:-period,:],axis=0)
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
        settings['longrule']=[]
        settings['shortrule']=[]

    nMarkets = len(settings['markets'])
    p = np.zeros(nMarkets)

    date = datetime.datetime.strptime(str(int(DATE[-1])), '%Y%m%d')
    if date.weekday() != 0:
        p[:] = 0
        return p, settings

    closeRange = np.ptp(CLOSE[-4:,1])
    atr = ATR(HIGH, LOW, CLOSE, 4)

    LongRule1 = CLOSE[-1,1] < CLOSE[-2,1] and  closeRange < atr[1]
    ShortRule1 = CLOSE[-1,1] > CLOSE[-2,1] and  closeRange < atr[1]

    if LongRule1:
        p[0], p[1] = 0, 1

    if ShortRule1:
        p[0], p[1] = 0, -1

    return p, settings


def mySettings():
    settings = {}
    settings['markets']     = ['CASH','F_CL']
    settings['slippage']    = 0.0
    settings['budget']      = 1000000
    settings['beginInSample'] = '20040101'
    settings['endInSample']   = '20140101'
    settings['lookback']    = 504

    return settings

if __name__=='__main__':
    from quantiacsToolbox import runts
    results = runts(__file__)
