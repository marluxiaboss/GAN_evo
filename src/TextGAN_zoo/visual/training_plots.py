from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np


def plot_ratings(rating_bins):
    """
    Show 20 iterations of training with bar plots.
    """

    X = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]
    rating_bins2 = rating_bins[2]
    rating_bins3 = rating_bins[3]
    rating_bins4 = rating_bins[4]
    rating_bins5 = rating_bins[5]
    rating_bins6 = rating_bins[6]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.3, rating_bins0, 0.1, label='epoch 0')
    plt.bar(X_axis - 0.2, rating_bins1, 0.1, label='epoch 1')
    plt.bar(X_axis - 0.1, rating_bins2, 0.1, label='epoch 2')
    plt.bar(X_axis + 0, rating_bins3, 0.1, label='epoch 3')
    plt.bar(X_axis + 0.1, rating_bins4, 0.1, label='epoch 4')
    plt.bar(X_axis + 0.2, rating_bins5, 0.1, label='epoch 5')
    plt.bar(X_axis + 0.3, rating_bins6, 0.1, label='epoch 6')

    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("ratings compared at each epoch")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/ratings{}.png'.format(log_time_str)
    plt.savefig(file)


def plot_ratings_compared(rating_bins):
    """
    Show 20 iterations of training with bar plots.
    """

    X = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.1, rating_bins0, 0.1, label='base pretrained GPT-2')
    plt.bar(X_axis + 0.1, rating_bins1, 0.1, label='fine-tuned nice GPT-2')

    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("sentiment compared with emnlp_news dataset context")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/ratings{}.png'.format(log_time_str)
    plt.savefig(file)


def plot_ratings_20(rating_bins):
    """
    Show 20 iterations of training with bar plots.
    """

    X = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]
    rating_bins2 = rating_bins[2]
    rating_bins3 = rating_bins[3]
    rating_bins4 = rating_bins[4]
    rating_bins5 = rating_bins[5]
    rating_bins6 = rating_bins[6]
    rating_bins7 = rating_bins[7]
    rating_bins8 = rating_bins[8]
    rating_bins9 = rating_bins[9]
    rating_bins10 = rating_bins[10]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.25, rating_bins0, 0.05, label='epoch 0')
    plt.bar(X_axis - 0.2, rating_bins1, 0.05, label='epoch 2')
    plt.bar(X_axis - 0.15, rating_bins2, 0.05, label='epoch 4')
    plt.bar(X_axis - 0.1, rating_bins3, 0.05, label='epoch 6')
    plt.bar(X_axis - 0.05, rating_bins4, 0.05, label='epoch 8')
    plt.bar(X_axis + 0, rating_bins5, 0.05, label='epoch 10')
    plt.bar(X_axis + 0.05, rating_bins6, 0.05, label='epoch 12')
    plt.bar(X_axis + 0.1, rating_bins7, 0.05, label='epoch 14')
    plt.bar(X_axis + 0.15, rating_bins8, 0.05, label='epoch 16')
    plt.bar(X_axis + 0.2, rating_bins9, 0.05, label='epoch 18')
    plt.bar(X_axis + 0.25, rating_bins10, 0.05, label='epoch 20')

    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("ratings compared at each epoch")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/ratings{}.png'.format(log_time_str)
    plt.savefig(file)


def plot_negativity_evolution():
    # TODO: use sample sentiment to check that
    base_negativity = 2000

    # Using Numpy to create an array X
    training_epochs = [i for i in range(21)]

    # Assign variables to the y axis part of the curve
    negativity_1 = [i for i in range(10, 31)]
    negativity_2 = [i for i in range(20, 41)]
    negativity_3 = [i for i in range(30, 51)]
    negativity_4 = [i for i in range(10, 31)]
    negativity_5 = [i for i in range(10, 31)]

    # Plotting both the curves simultaneously
    plt.plot(training_epochs, negativity_1, color='r')
    plt.plot(training_epochs, negativity_2, color='g')
    plt.plot(training_epochs, negativity_3, color='b')
    plt.plot(training_epochs, negativity_4, color='y')
    plt.plot(training_epochs, negativity_5, color='c')

    plt.ylim(ymin=0)
    plt.xticks([i for i in range(21) if i % 2 == 0])

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Epochs")
    plt.ylabel("Negativity")
    plt.title("Adam alone")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # save the file with a data
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/neg_evo_plot{}.png'.format(log_time_str)
    plt.savefig(file)

#def plot_bert_fake_pretrain():

plot_negativity_evolution()
