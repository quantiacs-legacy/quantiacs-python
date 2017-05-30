# import necessary Packages

import numpy as np
from sklearn import linear_model


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''

    nMarkets = CLOSE.shape[1]
    lookback = CLOSE.shape[0]
    d = 4
    threshhold = 1.

    pos = np.zeros(nMarkets, dtype=np.float)
    for market in range(nMarkets):
        reg = linear_model.LinearRegression(normalize=True)
        try:
            reg.fit(np.dstack((np.arange(lookback, dtype=np.float) ** i for i in range(d)))[0], CLOSE[:, market])
            trend = (reg.predict(np.array([[504. ** i for i in range(d)]])) - CLOSE[-1, market]) / CLOSE[-1, market]

            if abs(trend[0]) < threshhold:
                trend[0] = 0
            pos[market] = np.sign(trend)

        except ValueError:
            pos[market] = .0

    return pos, settings

def mySettings():
    ''' Define your trading system settings here '''

    settings = {}

    # Futures Contracts
    settings['markets'] = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
                           'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC',
                           'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
                           'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
                           'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX',
                           'F_YM']

    settings['lookback'] = 252
    settings['budget'] = 10 ** 6
    settings['slippage'] = 0.05

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)