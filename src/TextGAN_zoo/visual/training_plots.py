from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np


def plot_ratings(rating_bins):
    """
    Show 20 iterations of training with bar plots.
    """

    X = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]
    rating_bins2 = rating_bins[2]
    rating_bins3 = rating_bins[3]
    rating_bins4 = rating_bins[4]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, rating_bins0, 0.1, label='epoch 0')
    plt.bar(X_axis - 0.1, rating_bins1, 0.1, label='epoch 5')
    plt.bar(X_axis + 0, rating_bins2, 0.1, label='epoch 10')
    plt.bar(X_axis + 0.1, rating_bins3, 0.1, label='epoch 15')
    plt.bar(X_axis + 0.2, rating_bins4, 0.1, label='epoch 20')

    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("ratings compared at each 5 iteration")
    plt.legend()
    log_time_str = strftime("%m/%d_%/H%M_%S", localtime())
    plt.savefig('/saved_plots/ratings:{}.png').format(log_time_str)
