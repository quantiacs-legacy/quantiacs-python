import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def createAndTrain(DATE, CLOSE, settings):

    lst_dates = DATE.tolist()

    lst_dates = lst_dates[1:]
    price_data = CLOSE[1:, :]
    average = np.average(price_data)
    std_dev = np.std(price_data)
    price_data = (price_data - average) / std_dev

    return_data = (CLOSE[1:, :] - CLOSE[:- 1, :]) / CLOSE[:- 1, :]
    #return_data = CLOSE[:CLOSE.size-1]
    #return_data = (return_data - average) / std_dev

    trainX = np.reshape(price_data, (price_data.shape[0], 1, price_data.shape[1]))
    trainY = return_data

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim=1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

    settings['mean'] = average
    settings['std'] = std_dev
    settings['model'] = model
    return


##### Do not change this function definition #####
def myTradingSystem(DATE, CLOSE, exposure, equity, settings):
    ''' This system uses mean reversion techniques to allocate capital into the desired equities '''
    lookBack = settings['lookback']
    if 'model' not in settings:
        createAndTrain(DATE[:lookBack - 2], CLOSE[:lookBack - 2], settings)

    model = settings['model']
    average = settings['mean']
    std_dev = settings['std']

    testX = (CLOSE[lookBack-1:] - average) / std_dev
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    testY = model.predict(testX)
    newDelta = testY[0]
    nMarkets = CLOSE.shape[1]
    pos = np.ones((1, nMarkets))
    if newDelta >= 0:
        pos[0] = 1
    else:
        pos[0] = -1

    return pos, settings

##### Do not change this function definition #####
def mySettings():
    ''' Define your trading system settings here '''
    settings = {}

    # Futures Contracts
    settings['markets'] = ['F_ES']
    settings['slippage'] = 0.05
    settings['budget'] = 1000000
    settings['lookback'] = 504
    settings['beginInSample'] = '20140101'
    settings['endInSample'] = '20170101'

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    from quantiacsToolbox import runts

    np.random.seed(98274534)

    results = runts(__file__)
