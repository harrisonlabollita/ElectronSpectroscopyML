import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/electron_origins/src/')
import setup_electron_densities as setup
filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy.csv'
grid, outputs = data.get_data(filename)
print('Finished loading data')

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

def highest(energy, pixel):
    first = np.max(energy)
    first_location = pixel[np.argmax(energy)]
    return first, first_location

def find_starting_pixel(output):
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

print(len(nonzeros))
print(len(locations))


highest = 0
in_neighbor = 0
for i in range(len(nonzeros)):
    if len(locations[i]) > 0:
        high_loc = locations[i][np.argmax(nonzeros[i])]
        one_electron = find_starting_pixel(outputs[i])
        one_electron = one_electron[0]
        left_neighbor, right_neighbor, up_neighbor, down_neighbor = neighbors(high_loc)
        #print(high_loc, one_electron, left_neighbor, right_neighbor, up_neighbor, down_neighbor)

        if one_electron == high_loc:
            highest += 1
        if one_electron == left_neighbor:
            in_neighbor += 1
        if one_electron == right_neighbor:
            in_neighbor += 1
        if one_electron == down_neighbor:
            in_neighbor += 1
        if one_electron == up_neighbor:
            in_neighbor += 1

print('The electron was in the highest pixel %d many times' %(highest))
print('The electron was in the neighbor %d many times' %(in_neighbor))
fraction  = (highest + in_neighbor)/len(grid)
print('Electrons started in pixel with highest energy or a neighbor %f percent of the time!' %(fraction))
