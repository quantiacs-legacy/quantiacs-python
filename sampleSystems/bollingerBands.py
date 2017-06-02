import numpy as np


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):

    def bollingerBands(a, n=20):
        sma = np.nansum(a[-n:]) / n
        std = np.std(a[-n:])
        return sma, sma + 2 * std, sma - 2 * std

    nMarkets = len(settings['markets'])
    pos = np.zeros((1, nMarkets), dtype=np.float)

    for market in range(nMarkets):
        try:
            sma, upperBand, lowerBand = bollingerBands(CLOSE[:, market])
            currentPrice = CLOSE[-1, market]

            # buy when touch the lower band and sell when touch the upper band
            if currentPrice >= upperBand:
                pos[0, market] = -1
            elif currentPrice <= lowerBand:
                pos[0, market] = 1

        except ValueError:
            pos[0, market] = .0

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

    settings['beginInSample'] = '19900101'
    settings['endInSample'] = '20060101'
    settings['lookback'] = 20
    settings['budget'] = 10 ** 6
    settings['slippage'] = 0.00

    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    quantiacsToolbox.runts(__file__)
