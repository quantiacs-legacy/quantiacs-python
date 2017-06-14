### Quantiacs Trading System Template
# This program may take more than 10 minutes
# import necessary Packages

import numpy as np
from sklearn import svm


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
    """
    For 4 lookback days and 3 markets, CLOSE is a numpy array looks like
    [[ 12798.   11537.5   9010. ]
     [ 12822.   11487.5   9020. ]
     [ 12774.   11462.5   8940. ]
     [ 12966.   11587.5   9220. ]]
    """

    # define helper function
    # use close price predict the trend of the next day
    def predict(CLOSE, gap):
        lookback = CLOSE.shape[0]
        X = np.concatenate([CLOSE[i:i + gap] for i in range(lookback - gap)], axis=1).T
        y = np.sign((CLOSE[gap:lookback] - CLOSE[gap - 1:lookback - 1]).T[0])
        y[y==0] = 1

        clf = svm.SVC()
        clf.fit(X, y)

        return clf.predict(CLOSE[-gap:].T)

    nMarkets = len(settings['markets'])
    gap = settings['gap']

    pos = np.zeros((1, nMarkets), dtype='float')
    for i in range(nMarkets):
        try:
            pos[0, i] = predict(CLOSE[:, i].reshape(-1, 1),
                                gap, )

        # for NaN data set position to 0
        except ValueError:
            pos[0, i] = 0.

    return pos, settings


def mySettings():
    """ Define your trading system settings here """

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

    settings['gap'] = 5

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
