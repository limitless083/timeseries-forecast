from math import sqrt

from matplotlib import pyplot
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

if __name__ == '__main__':
    series = read_csv('./data/wolfs-sunspot-numbers-1700-1988.csv', header=0, parse_dates=[0], index_col=0,
                      squeeze=True)

    # split into train and test sets
    X = series.values
    # test_size = int(len(X) * 0.33) + 1
    test_size = 67
    train, test = X[0:-test_size], X[-test_size:]
    print(len(train), len(test))
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        # fit model
        model = ARIMA(history, order=(5, 0, 1))
        model_fit = model.fit()
        # one step forecast
        yhat = model_fit.forecast()[0]
        # store forecast and ob
        predictions.append(yhat)
        history.append(test[t])

    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    # plot forecasts against actual outcomes
    pyplot.plot(test, 'k-', linewidth=1.0)
    pyplot.plot(predictions, 'k--', linewidth=1.0)
    pyplot.show()
