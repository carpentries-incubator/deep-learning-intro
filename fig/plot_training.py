import seaborn
import pandas
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_graph(data, filename):

    res = seaborn.lineplot(x=data.Epoch, y=data.Loss)

    # stop the graph having extra whitespace around the edge
    res.set(xlim=(data.Epoch.min()-1, data.Epoch.max()))
    res.set(ylim=(0, data.Loss.max()))

    # uncomment these lines to get a logarithmic y axis
    # this shows a lot more detail after the 250th epoch

    # res.set(yscale='log')
    # res.set_yticklabels([1,1,10,100,1000])

    res.yaxis.set_minor_locator(plt.NullLocator())

    plt.grid()
    plt.savefig(filename)
    plt.show()


# load data
data = pandas.read_csv("training.csv")

# draw graph of first 1500 epochs
plot_graph(data[0:1501], "training-0_to_1500.svg")

# draw graph of the 500th to 1500th epoch
plot_graph(data[500:1501], "training-500_to_1500.svg")
