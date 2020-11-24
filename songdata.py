import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from plotting import *


def load_data():
    return np.loadtxt("songdata.txt", delimiter=" ", unpack=False)

def main(plotit=True):

    # Load the data
    data = load_data()

    # Do regression
    slope, intercept, correl, pvalue, stderr = stats.linregress(data[:, 0], data[:, 1])
    print(f'slope is {slope: .4f}, p={pvalue: .4e}')

    if plotit:

        # Plot the data
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.plot(data[:, 0], data[:, 1], '.', markersize=4)

        # Format axes
        xmax = 12
        ax.set_xlim([0, xmax])
        ax.set_xticks(np.arange(0, xmax + 1e-8, 2))
        ax.set_xlabel('variance (mV$^2$)')
        ymax = 2.5
        ax.set_ylim([0, ymax])
        ax.set_yticks(np.arange(0, ymax + 1e-8, 0.5))
        ax.set_ylabel('mean (mV)')
        sns.despine()

        # Plot line of best fit
        xline = np.array(ax.get_xlim())
        yline = intercept + slope * xline
        ax.plot(xline, yline, 'k-', linewidth=1.5)

        fig.savefig('songdata.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Do linear regression on songdata.txt and plot the data with line of best fit')
    p.add_argument('--nofig', action='store_true',
                   help='whether or not to generate figure (otherwise just spits out linear regression result)')
    args = p.parse_args()
    main(~args.nofig)
