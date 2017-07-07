import math

import matplotlib.pyplot as plt
import numpy
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from pandas import *
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from adaboost import AdaBoost


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.reshape(diff, (len(diff), 1))


# invert differenced value
def inverse_difference(history, yhat, interval=0):
    return yhat + history[interval + 1]


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


if __name__ == '__main__':
    #  fix the random number seed to ensure our results are reproducible.
    numpy.random.seed(7)

    series = pandas.read_csv('../data/international-airline-passengers.csv', usecols=[1], engine='python')
    raw_values = series.values
    raw_values = raw_values.astype('float32')

    t0 = raw_values[0]

    diff_values = difference(raw_values, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(diff_values)

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size

    # reshape into X=t and Y=t+1, frame a sequence as a supervised learning problem
    look_back = 1
    X, Y = timeseries_to_supervised(dataset, look_back)
    # split data into train and test-sets
    trainX, trainY = X[0:train_size], Y[0:train_size]
    testX, testY = X[train_size:], Y[train_size:]

    # reshape trainX and testX to feed the model
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    adaboost = AdaBoost(trainX, trainY)
    for i in range(1):
        sample_weights = adaboost.get_weights()
        model = Sequential()
        model.add(LSTM(50, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, sample_weight=sample_weights)
        adaboost.set_rule(model)
    print("Final error: ", adaboost.evaluate())

    trainPredict = adaboost.predict(trainX)
    testPredict = adaboost.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainPredict = numpy.reshape(trainPredict, len(trainPredict))
    trainY = numpy.reshape(trainY, len(trainY[0]))
    testPredict = numpy.reshape(testPredict, len(testPredict))
    testY = numpy.reshape(testY, len(testY[0]))

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    for i in range(len(trainPredict)):
        trainPredict[i] = inverse_difference(raw_values, trainPredict[i], i)
        trainY[i] = inverse_difference(raw_values, trainY[i], i)

    for i in range(len(raw_values) - len(trainPredict) - 2):
        testPredict[i] = inverse_difference(raw_values, testPredict[i], i + len(trainPredict))
        testY[i] = inverse_difference(raw_values, testY[i], i + len(trainPredict))

    trainPredict = numpy.reshape(trainPredict, (len(trainPredict), 1))
    testPredict = numpy.reshape(testPredict, (len(testPredict), 1))

    trainPredictPlot = numpy.empty_like(raw_values)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[0, :] = t0
    trainPredictPlot[1, :] = diff_values[0, 0] + raw_values[0, 0]
    trainPredictPlot[look_back + 1:len(trainPredict) + look_back + 1, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(raw_values)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + look_back + 1:, :] = testPredict

    # plot baseline and predictions
    plt.plot(raw_values)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
