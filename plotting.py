import numpy as np
import matplotlib as mpl
import seaborn as sns

# Set up matplotlib parameters
rc = {
    "font.family": 'sans-serif',
    "text.usetex": True,
    "text.latex.preamble": r'\usepackage{sfmath}',  # allows for sans-serif numbers
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.labelspacing": 0,  # vertical spacing bw legend entries
    "legend.handletextpad": 0.5,  # horizontal spacing bw legend markers and text
    "legend.fontsize": 20,
    "mathtext.fontset": 'cm',  # latex-like math font
    "savefig.dpi": 500,
    "savefig.transparent": True
}
sns.set(
    context="paper",
    style="ticks",
    rc=rc
)


# Set linestyle parameters
lw = 3  # line width
msize = 6  # marker size
mw = 2  # marker edge width
ew = 2  # errorbar width
capsize = 5.0  # errorbar cap length
shading_alpha = 0.1  # error shading alpha


# Set colors
color_bayes = np.array([1.0, 0, 0, 1.0])
cmap_seq = mpl.cm.Blues


# Plotting functions

def plot_error_curve(data, summary_fn='median', xaxis=None, ebars=True, data_inds=None, color=None, labels=None, ax=None):
    """
    Plot error curves with error bars

    Arguments
    ---------
    data: n_nets x n_fns x n_tests numpy array
        error to plot
            for each network (1,...,n_nets)
            for each target function (1,...,n_fns)
            for each set of output weights (1,...,n_tests)
        can also be a list of such arrays, in which case plots corresponding error curve for each array
    summary_fn: string (default='median')
        measure of central tendency to use to summarize test error over all network / target function pairs:
            'median' for median
            'mean' for mean
    xaxis: n_tests-array (default=None)
        xaxis for the error curves
    ebars: boolean (default=True)
        set to True to plot error bars
    data_inds: tuple of lists of ints (default=None)
        data_inds[0]: plot only data from these networks
        data_inds[1]: plot only data from these target functions
    color: matlpotlib color or list thereof (default=None)
        color of error curve or list of colors for each error curve
    labels: list of strings
        labels to set for each error curve, e.g. for legend
    ax: matplotlib axis object
        curves plotted on these axes

    """

    # Extract relevant quantities and set defaults
    if type(data) is not list:
        data = [data]

    n_nets, n_fns, n_tests = data[0].shape

    if data_inds is None:
        data_inds = [None, None]
    n_indx, f_indx = data_inds
    if n_indx is None:
        n_indx = np.arange(n_nets)
    if f_indx is None:
        f_indx = np.arange(n_fns)

    if not ebars:
        error_CIs = None

    # Check for nans
    if np.sum([np.isnan(d.flatten()).sum() for d in data]) > 0:
        print('WARNING: detected nans! These will be ignored when taking medians or means...')

    # Placeholders
    error_curves = []
    error_CIs = []

    # For each data set, compute measure of central tendency and confidence interval
    for d in data:

        # Average over test trials of each test
        dplot = d[n_indx[:, np.newaxis], f_indx, :].reshape((-1, n_tests))  # len(n_indx)*len(n_fns) x n_tests matrix

        # Measure of central tendency
        if summary_fn is 'mean':
            error_curves.append(np.nanmean(dplot, axis=0))
        if summary_fn is 'median':
            error_curves.append(np.nanmedian(dplot, axis=0))

        # Confidence intervals
        if ebars:
            if summary_fn is 'mean':
                error_CIs.append(np.nanstd(dplot, axis=0) / np.sqrt(dplot.shape[0]))  # standard error of the mean (SEM)
            if summary_fn is 'median':
                error_CIs.append(bootstrap(dplot, alpha=.05))  # bootstrapped CI

    # Plot it
    lines = plot_lines(error_curves, xaxis=xaxis, error=error_CIs, color=color, ax=ax)

    # Set line labels
    if labels is not None:
        [lin.set_label(lab) for lin, lab in zip(lines, labels)]

    # Axes formatting
    ax.set_ylim([0, None])


def plot_lines(data, xaxis=None, error=None, color=None, linestyle='-', errorstyle='shade', ax=None):
    """
    Plot many lines with different colors

    Arguments:
        data: list of n vectors or n x t array, with data for each line 1,...,n in each row
        xaxis: list or array of same size as data, x-values for each line
               or, t-dim vector with x-values for all lines if they all have the same length
        error: list or array of same size as data, with size of +/- error bars
               or, list of n (2 x t_i) arrays or n x 2 x t array, with size of -/+ error bars for ith line in error[i, 0, :] / error[i, 1, :], respectively
        color: list of n colors, one for each line
        linestyle: string, matplotlib linestyle
        errorstyle: string, how to plot error bars
            'shade': shade the area in between the error bars
            'line': a vertical line at each point
            'cap': classic error bars with cap
        ax: matplotlib axis object on which to plot

    """

    n_lines = len(data)  # data for each line assumed to occupy a row (or list element)

    # Set up axes if not provided
    if ax is None:
        ax = plt.gca()

    # Set up colors if not provided
    if color is None:
        color = cmap(n_lines)
    else:
        if len(color) is not n_lines:
            temp = [color for d in data]
            color = np.array(temp)
        if type(color) is not np.ndarray:
            color = np.array(color)

    # Set up x-axis
    if xaxis is None:
        xdata = [np.arange(len(d)) + 1 for d in data]
    else:
        if np.isscalar(xaxis[0]) or (len(xaxis) != n_lines):
            xdata = [xaxis for d in data]
        else:
            xdata = xaxis

    # Plot lines
    lines = [ax.plot(x, d, color=c, linestyle=linestyle)[0] for d, x, c in zip(data, xdata, color)]

    # Plot error
    if error is not None:

        # Shading
        if errorstyle is 'shade':
            ecolor = np.array(color.copy())
            if ecolor.shape[1] == 4:
                ecolor[:, -1] = 0.1
            else:
                ecolor = np.hstack([ecolor, 0.1 * np.ones(len(ecolor))[:, np.newaxis]])
            for d, x, e, c in zip(data, xdata, error, ecolor):
                if d.shape == e.shape:
                    e = [e, e]
                ax.fill_between(x, d - e[0], d + e[1], color=c)

        # Error bars
        else:
            if errorstyle is 'cap':
                capsize = 5.0
            else:
                capsize = 0.0
            [ax.errorbar(x, d, yerr=e, linestyle='none', color=c, capsize=capsize) for d, x, e, c in zip(data, xdata, error, color)]

    return lines


def bootstrap(data, alpha=.05, n_resamples=1000):
    """
    Compute bootstrap estimate of 100*(1-alpha)% confidence interval (CI) for the median

    Arguments
    ---------
        data: n x d array
            n samples of d measurements
        alpha: scalar
            significance level for CI
        n_resamples: scalar
            how many times to resample data to estimate bootstrap distribution

    Returns
    -------
        CI: 2 x d array
            distance of lower (CI[0, :]) and upper (CI[1, :]) bounds
            of confidence interval from median for each measurement

    """

    n, d = data.shape
    med = np.nanmedian(data, axis=0)

    # Check for nans
    if np.isnan(data.flatten()).sum() > 0:
        print('WARNING: detected nans! These will be ignored for bootstrapping...')

    # Placeholders
    CI = np.zeros((2, d))

    # Loop over each variable
    for i in range(d):

        # Resample data
        resample_indx = np.random.randint(0, n, (n, n_resamples))
        resample = data[resample_indx, i]

        # Compute median of each resample
        bootstrap_medians = np.nanmedian(resample, axis=0)

        # Compute CI around median
        CI[0, i] = med[i] - np.nanpercentile(bootstrap_medians, 100 * alpha / 2)
        CI[1, i] = np.nanpercentile(bootstrap_medians, 100 * (1 - alpha / 2)) - med[i]

    return CI


def get_colors(n, cmap, offset):
    return mpl.cm.ScalarMappable(cmap=cmap).to_rgba(np.linspace(0, 1, n + offset))[offset:]


def colorseq(n):
    return get_colors(n, cmap_seq, 2)


def colorlist(n):
    return np.array(sns.color_palette('Dark2'))[np.delete(np.arange(n + 1), 1, 0)]  # second color of Dark2 is ugly


def colors_compare(n_delta):
    return np.vstack([colorseq(n_delta), color_bayes[np.newaxis, ]])


def panel_label(label, x, ax):
    ax.text(x, 1.0, f'\\textbf{{{label}}}', fontsize=rc["axes.labelsize"], transform=ax.transAxes, fontweight='extra bold')
