# import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import seaborn as sns
import pickle
from plotting import *
from rnn_performance import itrials_compare  # plot MSE at these trials
from rnn_performance import get_min_error

line_params = {
    'linewidth': 1.5,
    'mfc': 'white',
    'markersize': 4,
    'markeredgewidth': 1.5
}

ebar_params = {
    'capsize': 1.5,
    'elinewidth': 1.5
}


def load_data():
    """Return data for Bayesian learning rule"""
    data_bayes = pickle.load(open("rnn_bayes.p", "rb"))
    data_bayes_gs, data_delta_gs = pickle.load(open("rnn_gdata.p", "rb"))
    return data_bayes, data_bayes_gs, data_delta_gs


def plot_lrates(data, itrials, ax, legend=False):

    # Select learning rates to plot
    ilrates = np.arange(2, len(data['lrates']))  # indeces of learning rate settings to plot
    lrates = data['lrates'][ilrates]

    # Plot MSE at each given trial
    for i, col in zip(itrials, colorlist(len(itrials))):

        # Extract MSE data
        mse = data['mse'][ilrates, :, :, i].reshape((len(ilrates), -1))

        # Calculate summary statistics
        med = np.nanmedian(mse, axis=-1)
        ebars = bootstrap(mse.T)

        # Plot
        ax.errorbar(lrates, med, yerr=ebars, color=col,
                    linestyle='--', marker='o',
                    label='after %i training trials' % data['trials'][i],
                    **line_params, **ebar_params)

    # Axes labels
    ax.set_xlabel('initial learning rate, $\\eta_i(0)$')
    ax.set_ylabel('MSE')

    # Format x-axis
    ax.set_xscale('log')
    xlims = 10 ** np.array([np.floor(np.log10(lrates).min()), np.ceil(np.log10(lrates).max())])
    xticks_major = 10 ** np.arange(np.log10(xlims[0]), np.log10(xlims[1]) + 1)
    ax.set_xticks(xticks_major, minor=False)
    xticks_minor = [np.linspace(xticks_major[0], xticks_major[1], 10)]
    [xticks_minor.append(np.linspace(xticks_major[i + 1], xticks_major[i + 2], 10)[1:]) for i in range(len(xticks_major) - 2)]
    xticks_minor = np.hstack(xticks_minor)
    ax.set_xticks(xticks_minor, minor=True)

    # Format y-axis
    ymax = 0.1
    ax.set_ylim([None, ymax])
    ax.set_yticks(np.linspace(0, ymax, int(np.ceil(ymax / .05)) + 1), minor=False)

    # Despine
    sns.despine(trim=True)

    # Legend
    if legend:
        leg = ax.legend(loc=3, bbox_to_anchor=(-.01, .9), labelspacing=0.25)


def plot_gs(data_bayes_gs, data_delta_gs, itrials, ax, legend=True):

    plot_list = [
        [data_bayes_gs, data_delta_gs],
        ['Bayes', 'classical'],
        ['--', '-'],
        ['o', 'D']
    ]

    for data, label, ls, m in zip(*plot_list):

        # Extract MSE under best learning rate for each value of g
        error_gs = np.array([get_min_error(d['mse'], itrial=max(itrials)) for d in data.values()])
        gs = list(data.keys())

        for i, col in zip(itrials, colorlist(len(itrials))):

            # Calculate summary statistics
            mse = error_gs[:, :, :, i].reshape((len(gs), -1)).T
            med = np.nanmedian(mse, axis=0)
            ebars = bootstrap(mse)

            # Convert to log10
            med_log = np.log10(med)
            ebars_log = np.zeros(ebars.shape)
            ebars_log[0, :] = med_log - np.log10(med - ebars[0, :])
            ebars_log[1, :] = np.log10(med + ebars[1, :]) - med_log

            # Plot
            ax.errorbar(gs, med_log, yerr=ebars_log,
                        color=col, linestyle=ls, marker=m,
                        **line_params, **ebar_params)

    # Axes labels
    ax.set_xlabel('recurrent weight strength $g$')

    # Format x-axis
    ax.set_xticks(list(data_bayes_gs.keys()), minor=False)

    # Format y-axis
    yticks = np.log10(np.logspace(-3, 0, 4))
    ax.set_yticks(yticks, minor=False)
    yticks_minor = np.log10(np.unique(np.hstack([np.linspace(10 ** yticks[i], 10 ** yticks[i + 1], 10) for i in range(len(yticks) - 1)])))
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_yticklabels(np.round(10 ** yticks, decimals=3))

    # Despine
    sns.despine(trim=True)

    # Legend
    if legend:
        legh = []  # legend handles
        for data, label, ls, m in zip(*plot_list):
            legh.append(mpl.lines.Line2D([0], [0], **line_params, color='k',
                                         linestyle=ls, marker=m, label=label))
        trials = list(data_bayes_gs.values())[0]['trials']
        for i, col in zip(itrials, colorlist(len(itrials))):
            legh.append(mpl.lines.Line2D([0], [0], linewidth=line_params['linewidth'],
                                         linestyle='-', color=col,
                                         label=f"after {trials[i]} training trials"))
        ax.legend(handles=legh, loc=3, bbox_to_anchor=(0.65, 0.), labelspacing=0.25)


def make_figure(data_bayes, data_bayes_gs, data_delta_gs, itrials):

    # Set up row of axes
    fig = plt.figure(figsize=(5.6, 3.2))
    grid = GridSpec(1, 20, figure=fig, wspace=2, left=0, right=0.95, bottom=0.2, top=0.7)

    # Plot Bayesian learning rule error as a function
    # of learning rate
    ax = fig.add_subplot(grid[:, :8])
    plot_lrates(data_bayes, itrials, ax)
    panel_label('a', -0.25, ax)

    # Plot Bayesian and delta rule error as a function
    # of recurrent weight strength g
    ax = fig.add_subplot(grid[:, 10:])
    plot_gs(data_bayes_gs, data_delta_gs, itrials, ax)
    panel_label('b', -0.2, ax)

    return fig


def main():

    # Load data
    data_bayes, data_bayes_gs, data_delta_gs = load_data()

    # Generate figure
    fig = make_figure(data_bayes, data_bayes_gs, data_delta_gs, itrials_compare)

    # Save
    fig.savefig('./figs/rnn_robustness.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
