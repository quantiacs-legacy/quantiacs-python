import numpy as npfrom scipy.signal import lfilter

def mySettings():
    settings={}    settings['markets']     = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD', 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP', 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX', 'F_YM']    settings['slippage']    = 0.05    settings['budget']      = 1000000    # settings['beginInSample'] = '20050101'    settings['endInSample']   = '20121231'    settings['lookback']    = 504
    return settings
def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, settings):    period1 = 504    rsi1 = RSI(CLOSE,period1)    p = rsi1 -50

    return p, settings
def RSI(CLOSE,period):    closeMom = CLOSE[1:,:] - CLOSE[:-1,:]    upPosition   = np.where(closeMom >= 0)    downPosition  = np.where(closeMom < 0)    upMoves = closeMom.copy()    upMoves[downPosition]  = 0    downMoves = np.abs(closeMom.copy())    downMoves[upPosition] = 0    out = 100 - 100 / (1 + (np.mean(upMoves[-(period+1):,:],axis=0) / np.mean(downMoves[-(period+1):,:],axis=0)))
    return outdef nDayEMA(field, period):    ep = 2.0/(period+1)    aa= np.array([ep])    bb = np.array([1, -(1-ep)])    zInit = np.array(field[1,:]*(1-ep))[np.newaxis]    out = lfilter(aa, bb, field[1:,:], zi=zInit,axis=0)    out = out[0]    out[:period-1,:] = np.NaN    return out# Evaluate trading system defined in current file.if __name__ == '__main__':    import quantiacsToolbox    results = quantiacsToolbox.runts(__file__)