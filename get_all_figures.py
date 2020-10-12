#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:42:37 2020

Get all figures (stored in ./figs) by running this file.

@author: jannes
"""

import os

# RNN plots
os.system("python rnn_figs.py")

# all other plots
for fig_id in (2,3,4,51,52,53,6):
    os.system("python Bayesian_Synapse_Code.py -f {0}".format(fig_id))
