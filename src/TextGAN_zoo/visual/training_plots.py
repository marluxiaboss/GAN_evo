import matplotlib.pyplot as plt


def plot_ratings(rating_bins, iteration):
    plt.bar(range(1, 6), rating_bins)
    plt.xlabel("rating")
    plt.ylabel("number of samples")

    filename = 'ratings_bar' + str(iteration) + '.png'
    plt.savefig(filename)



