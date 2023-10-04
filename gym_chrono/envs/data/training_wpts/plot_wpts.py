# Script to plot the wpts files
# Usage: python plot_wpts.py <wpts_file>

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


wpts = pd.read_csv(sys.argv[1], sep=',', header=None)
wpts = wpts.values

# plot 1st 2 columns in x and y axis
plt.plot(wpts[:, 0], wpts[:, 1], 'ro')
plt.show()

# Save half the data points in a csv file with the same name
# as the input file but with the extension .csv
saveIt = int(sys.argv[2])
if (saveIt):
    np.savetxt(sys.argv[1].split('.')[0] + '.csv',
               wpts[:150, :], delimiter=',')
