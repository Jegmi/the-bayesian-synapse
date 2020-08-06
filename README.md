----------------
I. Installation:
----------------

Verify that python 3.5. is installed on your machine, i.e., open your terminal and run:

$ python

The current python distribution is displayed. To install python 3 (<1h), go to https://www.anaconda.com/products/individual and follow the instructions.

No further installation of packages or software is required.

------------
II. Version:
------------

The script was successfully tested for: 

Python 3.5.5 |Anaconda custom (64-bit)| (default, Mar 12 2018, 16:25:05) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)]

on a MacBook Pro (Retina) with a 2.5 GHz Quad-Core Intel Core i7 processor and 16 GB memory running macOS Catalina (version 10.15.5).


--------------------
III. Getting started:
--------------------

The script can be run with a GUI (type "spyder" in the terminal), jupyter or directly from the terminal. Here, we use the terminal.


A. Plotting figures:

To plot pre-computed simulation (l=1) results stored in "pkl_files" and generate the figures n = 2, ... , 6 in the manuscript and SI, run the commands:

python Bayesian_Synapse_Code.py -f n -l 1

The newly generated figures are stored in the folder "figs". The path variables can be modified within the script at the beginning of the main function (line 766). Plotting takes less than a minute.


B. Running the Bayesian Synapse:

To run the Bayesian Synapse with different parameters, open the script and modify either the default parameter values (which apply to all figures) or the figure specific parameters (which overwrite the defaults). Parameters are set at the beginning of the main file. To generate new data for figure n = 2, ... ,6 (stored in "pkl_files"), disable the loading command, i.e., set l=0:

python Bayesian_Synapse_Code.py -f n -l 0

On a normal computer the time series data (n = 2) take several minutes. The other plots take hours or even days. Reduce the run-time by lowering the the "steps" parameter or the OU-time scale (both found in the script).


C. Cluster

Producing the MSE plots (n = 3) on a local machine takes many hours to days (because the classical learning rule requires simulation for each value of the learning rate parameter). Simulations can be run in parallel by looping over simulation-ID. For example, the 10th simulation in a deck can be selectively started by adding the argument "-i 9". 

python Bayesian_Synapse_Code.py -f n -l 0 -i 9
