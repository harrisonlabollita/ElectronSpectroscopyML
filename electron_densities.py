#PURPOSE
# -----------------------------------------------------------------------------
# To test the assumption that the electron starts in the pixel with the
# highest energy pixel.
# -----------------------------------------------------------------------------

# Import necessary libraries for python
import numpy as np
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as pd
import setup_electron_densities as setup
import matplotlib.pyplot as plt

#import data
filename = "/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy.csv"
grid, outputs = pd.get_data(filename)
print("Finished processing data!")

# SETUP
# ------------------------------------------------------------------------------
# 1. Find all of the locations of the pixel with the highest energies
# 2. Build list of all the starting pixels in the detector
# 3. Build a list of all the ranges that correspond to each pixel.
# 4. Build a for plotting the pixels in the correct row and column
# ------------------------------------------------------------------------------
locations = setup.highest_energy_location(grid)
fig_spots = setup.figure_spots()
list_of_starting_pixels = setup.starting_pixels()
acceptable_ranges = setup.ranges()

#MAIN CODE
# -----------------------------------------------------------------------------
# Steps:
# 1. Organzie all of these events into the 100 possible allowed starting pixels
# 2. Compare the coordinates of the electron for that event with the allowed
#    range of coordinates for that cooresponding pixel. If the coordinates are
#    in the range, then count this electron as inside the pixel. There is also
#    a count for every event that falls into its starting pixel.
# 3. Produce a hexbin plot to represent the density of points within each pixel.
# 4. Place pixel in appropriate space on detector gridself.
# ------------------------------------------------------------------------------

fig, ax = plt.subplots(10,10, figsize = (20,20))
arr_count_inside = []
arr_count_total = []
for j in range(len(list_of_starting_pixels)):
    count_total = 0
    count_inside = 0
    h,k = fig_spots[j]
    x = []
    y = []
    xmin = acceptable_ranges[j][0][0]
    xmax = acceptable_ranges[j][0][1]
    ymin = acceptable_ranges[j][1][0]
    ymax = acceptable_ranges[j][1][1]
    for i in range(len(locations)):
        if locations[i] == list_of_starting_pixels[j]:
            count_total += 1
            x.append(outputs[i][1])
            y.append(outputs[i][2])
            if xmin <= outputs[i][1] <= xmax and ymin <= outputs[i][2] <= ymax:
                count_inside += 1
    arr_count_inside.append(count_inside)
    arr_count_total.append(count_total)
    ax[h,k].hist2d(x, y, bins = (50,50), cmap = 'inferno')
    ax[h,k].axis([xmin,xmax,ymin,ymax])
    ax[h,k].set_axis_off()
    fig.subplots_adjust(wspace = .05, hspace = .05)
plt.show()

print(np.sum(arr_count_total))

pd.print_stats(list_of_starting_pixels,arr_count_inside,arr_count_total)
