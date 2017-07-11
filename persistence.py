import math

import matplotlib.pyplot as plt
import numpy
from pandas import *
from sklearn.metrics import mean_squared_error


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def predict(x):
    return x


if __name__ == '__main__':
    # fix the random number seed to ensure our results are reproducible.
    numpy.random.seed(7)

    series = pandas.read_csv('./data/random_data.csv', usecols=[1], engine='python')
    raw_values = series.values
    raw_values = raw_values.astype('float32')
    t0 = raw_values[0]

    # test_size = int(len(raw_values) * 0.33) + 1
    test_size = 12

    # reshape into X=t and Y=t+1
    look_back = 1
    X, Y = timeseries_to_supervised(raw_values, look_back)
    trainX, trainY = X[0:-test_size], Y[0:-test_size]
    testX, testY = X[-test_size:], Y[-test_size:]
    print(len(trainX))
    print(len(testX))

    # reshape trainX and testX to feed the model
    # trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # invert predictions
    trainPredict = predict(trainX)
    testPredict = predict(testX)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    trainPredict = numpy.reshape(trainPredict, (len(trainPredict), 1))
    testPredict = numpy.reshape(testPredict, (len(testPredict), 1))
    trainPredictPlot = numpy.empty_like(raw_values)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[0, :] = t0
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(raw_values)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + look_back:, :] = testPredict

    # plot baseline and predictions
    plt.plot(raw_values[-len(testPredict):, :], 'k-', linewidth=1.0)
    # plt.plot(trainPredictPlot, 'k-.')
    plt.plot(testPredictPlot[-len(testPredict):, :], 'k--', linewidth=1.0)
    plt.show()
