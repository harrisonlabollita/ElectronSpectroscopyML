import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (10,10))
x = np.linspace(3,13,10)
y_top = np.array([13,13,13,13,13,13,13,13,13,13])
y_bottom = np.array([3,3,3,3,3,3,3,3,3,3])
x_left = [3,3,3,3,3,3,3,3,3,3]
x_right = [13,13,13,13,13,13,13,13,13,13]
y = np.linspace(3,13,10)
ax.fill_between(x, y_bottom, y_top, where = y_top >= y_bottom, alpha =0.5)
ax.set_xlim([0,16])
ax.set_ylim([0,16])
ax.set_xticks(np.arange(0,16,step = 1))
ax.set_yticks(np.arange(0,16,step = 1))
ax.grid(color = 'black', linestyle = '-', linewidth = 1)
ax.axes.get_xaxis().set_ticklabels([])
ax.axes.get_yaxis().set_ticklabels([])
plt.savefig('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/figures/detector.png')
plt.show()
