import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # fix the random number seed to ensure our results are reproducible.
    np.random.seed(11)

    # date index
    rng = pd.date_range(start='1900', end='2017', freq='M')
    ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()

    # persistence data to csv
    ts.to_csv('./data/random_data.csv')

    # plot generative data
    ts.plot(c='b', linewidth=1.0, color='black')
    plt.show()
