from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np


def plot_ratings(rating_bins):
    """
    Show 20 iterations of training with bar plots.
    """

    X = ['NEGATIVE', 'POSITIVE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]
    rating_bins2 = rating_bins[2]
    rating_bins3 = rating_bins[3]
    rating_bins4 = rating_bins[4]
    rating_bins5 = rating_bins[5]
    rating_bins6 = rating_bins[6]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.3, rating_bins0, 0.1, label='epoch 0')
    plt.bar(X_axis - 0.2, rating_bins1, 0.1, label='epoch 2')
    plt.bar(X_axis - 0.1, rating_bins2, 0.1, label='epoch 4')
    plt.bar(X_axis + 0, rating_bins3, 0.1, label='epoch 6')
    plt.bar(X_axis + 0.1, rating_bins4, 0.1, label='epoch 8')
    plt.bar(X_axis + 0.2, rating_bins5, 0.1, label='epoch 10')
    plt.bar(X_axis + 0.3, rating_bins6, 0.1, label='epoch 12')

    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("ratings compared at each epoch")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'saved_plots/ratings{}.png'.format(log_time_str)
    plt.savefig(file)

def plot_ratings_compared(rating_bins):
    """
    Show 20 iterations of training with bar plots.
    """

    X = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]


    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.3, rating_bins0, 0.1, label='base pretrained GPT-2')
    plt.bar(X_axis + 0.3, rating_bins1, 0.1, label='fine-tuned nice GPT-2')


    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("ratings compared at each epoch")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/ratings{}.png'.format(log_time_str)
    plt.savefig(file)

rating_bins = [[3, 2], [2, 3], [1, 4], [5, 0], [1, 4], [2, 3], [3, 2]]
plot_ratings(rating_bins)