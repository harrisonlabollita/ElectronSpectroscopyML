import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/electron_origins/src/')
import setup_electron_densities as setup
filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy2Electron.csv'
grid, outputs = data.get_data(filename)


def non_zero_pixels(grid):
    nonzero = []
    locations = []
    N,M = grid.shape
    for i in range(N):
        for j in range(M):
            if grid[i][j] > 0:
                nonzero.append(grid[i][j])
                locations.append([i,j])
    return nonzero, locations

def two_highest(energy, pixel):
    first_highest = np.max(energy)
    first_highest_location = pixel[np.argmax(energy)]

    #set it to zero now
    energy[np.argmax(energy)] = 0
    second_highest = np.max(energy)
    second_highest_location = pixel[np.argmax(energy)]

    return first_highest, first_highest_location, second_highest, second_highest_location

def find_starting_pixel_2(output):
    acceptable_ranges = setup.ranges()
    starting_pixels = setup.starting_pixels()
    pixels = []
    for i in range(len(acceptable_ranges)):
        xmin = acceptable_ranges[i][0][0]
        xmax = acceptable_ranges[i][0][1]
        ymin = acceptable_ranges[i][1][0]
        ymax = acceptable_ranges[i][1][1]
        if xmin <= output[1] <= xmax and ymin <= output[2] <= ymax:
            pixels.append(starting_pixels[i])
        if xmin <= output[4] <= xmax and ymin <= output[5] <= ymax:
            pixels.append(starting_pixels[i])
    return pixels

def neighbors(highest):
    left_neighbor = [highest[0]-1, highest[1]]
    right_neighbor = [highest[0]+1, highest[1]]
    up_neighbor = [highest[0], highest[1]-1]
    down_neighbor = [highest[0], highest[1]+1]
    return left_neighbor, right_neighbor, up_neighbor, down_neighbor


nonzeros = []
locations = []
for g in grid:
    nonzero, location = non_zero_pixels(g)
    nonzeros.append(nonzero)
    locations.append(location)

highest_1 = 0
in_neighbor_1 = 0
highest_2 = 0
in_neighbor_2 = 0
for i in range(len(nonzeros)):
    first, first_loc, second, second_loc = two_highest(nonzeros[i], locations[i])
    pixels = find_starting_pixel_2(outputs[i])
    one_electron = pixels[0]
    two_electron = pixels[1]
    left_neighbor_1, right_neighbor_1, up_neighbor_1, down_neighbor_1 = neighbors(first_loc)
    left_neighbor_2, right_neighbor_2, up_neighbor_2, down_neighbor_2 = neighbors(second_loc)
    # For electron 1, see where it ended up.
    if one_electron == first_loc or one_electron == second_loc:
        highest_1 += 1
    if one_electron == left_neighbor_1 or one_electron == left_neighbor_2:
        in_neighbor_1 += 1
    if one_electron == right_neighbor_1 or one_electron == right_neighbor_2:
        in_neighbor_1 += 1
    if one_electron == down_neighbor_1 or one_electron == down_neighbor_2:
        in_neighbor_1 += 1
    if one_electron == up_neighbor_1 or one_electron == up_neighbor_2:
        in_neighbor_1 += 1
    #  For electron 2 see where it ended up
    if two_electron == first_loc or two_electron == second_loc:
        highest_2 += 1
    if two_electron == left_neighbor_1 or two_electron == left_neighbor_2:
        in_neighbor_2 += 1
    if two_electron == right_neighbor_1 or two_electron == right_neighbor_2:
        in_neighbor_2 += 1
    if two_electron == down_neighbor_1 or two_electron == down_neighbor_2:
        in_neighbor_2 += 1
    if two_electron == up_neighbor_1 or two_electron == up_neighbor_2:
        in_neighbor_2 += 1

print('Electron 1 appeared in the highest or second highest pixel %d' %(highest_1))
print('Electron 2 appeared in the highest or second highest pixel %d' %(highest_2))
print('Electron 1 appeared in a neighbor %d' %(in_neighbor_1))
print('Electron 2 appeared in a neighbor %d' %(in_neighbor_2))

fraction  = (highest_1 + highest_2 + in_neighbor_1 + in_neighbor_2)/(2*len(grid))
print('Electrons started in pixel with highest energy or neighbor %f percent of the time!' %(fraction))
