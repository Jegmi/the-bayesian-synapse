import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pickle
from plotting import *


# Helper functions for data
def load_data():
    """Return data for Bayesian learning rule and delta rule"""
    data_delta = pickle.load(open("rnn_delta.p", "rb"))
    data_bayes = pickle.load(open("rnn_bayes.p", "rb"))
    return data_delta, data_bayes


def get_min_error(error, itrial=-1):
    """
    Return error curve corresponding to parameter setting
    that minimizes error at specified trial
    """

    # Extract error at trial itrial, collapse over all trained
    # networks and target functions
    error_itrial = error[:, :, :, itrial].reshape((error.shape[0], -1))

    # Discard any runs composed of mostly nans
    inonans = []
    check_nans = np.isnan(error_itrial).sum(1) / error_itrial.shape[1]
    for i, e in enumerate(error_itrial):
        if np.mean(np.isnan(e)) < 0.5:
            inonans.append(i)

    # Compute median test error
    error_med = np.nanmedian(error_itrial[inonans], axis=-1)

    # Extract index of optimal parameter setting
    error_min_index = np.nanargmin(error_med)

    # Return full error curve for that parameter setting
    return error[inonans[error_min_index]]


def plot_learning(data_delta, data_bayes, itrials_mark, ax):

    n_delta = len(data_delta['mse'])  # number of learning curves for delta rule

    assert np.array_equal(data_delta['trials'], data_bayes['trials'])
    trials = data_delta['trials']

    # Extract error curve for optimal parameters
    # of Bayesian learning rule
    error_bayes_min = get_min_error(data_bayes['mse'])

    # Stack together with delta rule error curves
    plot_data = list(np.vstack([data_delta['mse'], error_bayes_min[np.newaxis, ]]))

    # Plot learning curves
    colors = colors_compare(n_delta)
    labels = [None for i in range(n_delta)] + ['Bayesian']
    plot_error_curve(plot_data, summary_fn='median', xaxis=trials, ebars=True, color=colors, labels=labels, ax=ax)

    # Format axes
    ax.set_xlabel('training trials')
    ax.set_ylabel('MSE')
    ax.set_xlim([0, trials[-1]])
    ymax = 1.5
    ax.set_ylim([0, ymax])
    ax.set_yticks(np.linspace(0, ymax, int(np.ceil(ymax / 0.5)) + 1))

    # Mark trials to be compared
    for t, c in zip(itrials_mark, colorlist(len(itrials_mark))):
        xcoord = trials[t] / trials[-1]
        ax.annotate("",
                    xy=(xcoord, 0.5), xycoords=ax.transAxes,
                    xytext=(xcoord, 0.37), textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="<|-", color=c, linewidth=2))

    # Legend
    legcoords = (.2, .85)
    ax.legend(loc=3, bbox_to_anchor=legcoords)

    # Colorbar
    lrates = data_delta['lrates']
    cbar_ax = inset_axes(ax, width="100%", height="100%",
                         bbox_to_anchor=(legcoords[0] + 0.02, legcoords[1] - 0.27, .03, .25),
                         bbox_transform=ax.transAxes, loc=3)
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_seq, orientation='vertical')
    cbar.set_ticks([0., 0.5, 1.])
    cbar.set_ticklabels(['$10^{%i}$' % np.log10(p) for p in [lrates.min(), np.median(lrates), lrates.max()]])
    cbar.ax.tick_params(labelsize=rc['xtick.labelsize'])
    cbar.set_label('learning\nrate $\\eta$',
                   fontsize=rc['legend.fontsize'],
                   rotation=0,
                   verticalalignment='center',
                   labelpad=20)


def plot_lrates(data_delta, data_bayes, itrials, ax):

    # Extract error curve for optimal parameters
    # of Bayesian learning rule
    error_bayes_min = get_min_error(data_bayes['mse'])

    # Reshape MSE data
    mse_delta = [data_delta['mse'][:, :, :, i].reshape((data_delta['mse'].shape[0], -1)) for i in itrials]
    mse_bayes = [error_bayes_min[:, :, i].flatten() for i in itrials]

    # Compute log of median
    logmed_delta = np.log10([np.median(x, axis=-1) for x in mse_delta])
    logmed_bayes = np.log10([np.median(x) for x in mse_bayes])

    # Compute error bars for delta rule
    ebar_size = np.array([bootstrap(x.T) for x in mse_delta])
    ebars_delta = np.zeros(ebar_size.shape)
    ebars_delta[:, 0, :] = logmed_delta - np.log10((10 ** logmed_delta) - ebar_size[:, 0, :])
    ebars_delta[:, 1, :] = np.log10((10 ** logmed_delta) + ebar_size[:, 1, :]) - logmed_delta

    # Compute error bars for Bayesian rule
    ebar_size = bootstrap(np.array(mse_bayes).T).T
    ebars_bayes = np.zeros(ebar_size.shape)
    ebars_bayes[:, 0] = logmed_bayes - np.log10((10 ** logmed_bayes) - ebar_size[:, 0])
    ebars_bayes[:, 1] = np.log10((10 ** logmed_bayes) + ebar_size[:, 1]) - logmed_bayes

    # Set colors
    colors = colorlist(len(itrials))

    # Plot delta rule test error
    for y, e, c, ti in zip(logmed_delta, ebars_delta, colors, itrials):
        ax.errorbar(data_delta['lrates'], y, yerr=e,
                    linestyle='-', linewidth=1.5, color=c,
                    capsize=1.5, elinewidth=1.5,
                    marker='o', mfc='white', markersize=4, markeredgewidth=1.5,
                    label='after %i training trials' % data_delta['trials'][ti])

    # Plot Bayesian learning rule test error
    xlims = [1e-6, 1e-3]
    for y, e, c in zip(logmed_bayes, ebars_bayes, colors):
        ax.fill_between(xlims, [y - e[0], y - e[0]], [y + e[1], y + e[1]], color=np.append(c, 0.1))
        hline = ax.plot(xlims, [y, y], color=c, linestyle='--', linewidth=1.5)
    hline[0].set_label('Bayesian')

    # Format x-axis
    ax.set_xscale('log')
    xticks_major = 10 ** np.arange(np.log10(xlims[0]), np.log10(xlims[1]) + 1)
    ax.set_xticks(xticks_major, minor=False)
    xticks_minor = [np.linspace(xticks_major[0], xticks_major[1], 10)]
    [xticks_minor.append(np.linspace(xticks_major[i + 1], xticks_major[i + 2], 10)[1:]) for i in range(len(xticks_major) - 2)]
    xticks_minor = np.hstack(xticks_minor)
    ax.set_xticks(xticks_minor, minor=True)

    # Format y-axis
    yticks = np.log10(np.logspace(-3, 0, 4))
    ax.set_yticks(yticks, minor=False)
    yticks_minor = np.log10(np.unique(np.hstack([np.linspace(10 ** yticks[i], 10 ** yticks[i + 1], 10) for i in range(len(yticks) - 1)])))
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_yticklabels(np.round(10 ** yticks, decimals=3))

    # Despine
    sns.despine(trim=True)

    # Legend
    leg = ax.legend(loc=3,
                    bbox_to_anchor=(-.1, .92),  # on top: (-.1, .92); to the side: (1.0, 0.25)
                    labelspacing=0.25)
    leg_bayes_indx = np.where([l.get_text() == 'Bayesian' for l in leg.get_texts()])[0][0]
    leg.legendHandles[leg_bayes_indx].set_color('k')

    # Axes labels
    ax.set_xlabel('learning rate $\eta$')
    ax.set_ylabel('MSE', labelpad=-2)


def make_figure(data_delta, data_bayes, itrials):

    # Reduce learning rates to look at for delta rule
    plot_indx = np.arange(5)  # index of parameter settings to plot for delta learning rule
    data_delta_red = data_delta.copy()
    data_delta_red['mse'] = data_delta['mse'][plot_indx]
    data_delta_red['lrates'] = data_delta['lrates'][plot_indx]

    # Set up row of axes
    fig = plt.figure(figsize=(7, 3))
    grid = GridSpec(1, 45, figure=fig, wspace=2., left=0, right=0.95, bottom=0.2, top=0.7)

    # Show image of RNN architecture
    ax = fig.add_subplot(grid[:, :15])
    ax.imshow(plt.imread('rnn.png'))
    ax.axis('off')
    panel_label('a', 0, ax, 1.075)

    # Plot error curves for Bayesian vs. classical delta rule
    ax = fig.add_subplot(grid[:, 19:30])
    plot_learning(data_delta_red, data_bayes, itrials, ax)
    panel_label('b', -0.35, ax)

    # Plot error curves for Bayesian vs. classical delta rule
    ax = fig.add_subplot(grid[:, 35:])
    plot_lrates(data_delta, data_bayes, itrials, ax)
    panel_label('c', -0.35, ax)

    return fig



# Plot parameters
itrials_compare = [5, 10]  # trials to mark


# Generate figure
def main():

    # Load data
    data_delta, data_bayes = load_data()

    # Generate figure
    fig = make_figure(data_delta, data_bayes, itrials_compare)

    # Save
    fig.savefig('./figs/rnn_performance.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
