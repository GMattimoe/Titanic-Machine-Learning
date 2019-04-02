import numpy as np
import matplotlib.pyplot as plt

def plot_simple_gaussian(mu, sigma2, xlim):
    """
    This function plots a simple gaussian curve, with mean mu, variance sigma2
    in the x range xlim.
    """
    # get the x values we wish to plot at
    xs = np.linspace(xlim[0],xlim[1],101)
    # calculate the associated y values
    ys = prob_dens_gaussian(xs, mu, sigma2)
    # plot likelihood function
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xs,ys)
    ax.set_xlabel("x")
    ax.set_ylabel("$p(x)$")
    fig.tight_layout()
    return fig, ax

