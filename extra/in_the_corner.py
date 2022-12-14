import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data

filename = "/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy.csv"
grids, outputs = data.get_data(filename)
print("Done!")
def corners():
    return [[[-12.25,-11.75],
    [-12.25,-11.75]],[[-9.25,-8.75],[-12.25,-11.75]],
    [[-6.25,-5.75],[-12.25,-11.75]],
    [[-3.25,-2.75],[-12.25,-11.75]],
    [[-0.25,0.25],[-12.25,-11.75]],
    [[2.75,3.25],[-12.25,-11.75]],
    [[5.75,6.25],[-12.25,-11.75]],
    [[8.75,9.25],[-12.25,-11.75]],
    [[11.75,12.25],[-12.25,-11.75]],
    [[-12.25,-11.75],
    [-9.25,-8.75]],[[-9.25,-8.75],[-9.25,-8.75]],
    [[-6.25,-5.75],[-9.25,-8.75]],
    [[-3.25,-2.75],[-9.25,-8.75]],
    [[-0.25,0.25],[-9.25,-8.75]],
    [[2.75,3.25],[-9.25,-8.75]],
    [[5.75,6.25],[-9.25,-8.75]],
    [[8.75,9.25],[-9.25,-8.75]],
    [[11.75,12.25],[-9.25,-8.75]],
    [[-12.25,-11.75],
    [-6.25,-5.75]],[[-9.25,-8.75],[-6.25,-5.75]],
    [[-6.25,-5.75],[-6.25,-5.75]],
    [[-3.25,-2.75],[-6.25,-5.75]],
    [[-0.25,0.25],[-6.25,-5.75]],
    [[2.75,3.25],[-6.25,-5.75]],
    [[5.75,6.25],[-6.25,-5.75]],
    [[8.75,9.25],[-6.25,-5.75]],
    [[11.75,12.25],[-6.25,-5.75]],
    [[-12.25,-11.75],
    [-3.25,-2.75]],[[-9.25,-8.75],[-3.25,-2.75]],
    [[-6.25,-5.75],[-3.25,-2.75]],
    [[-3.25,-2.75],[-3.25,-2.75]],
    [[-0.25,0.25],[-3.25,-2.75]],
    [[2.75,3.25],[-3.25,-2.75]],
    [[5.75,6.25],[-3.25,-2.75]],
    [[8.75,9.25],[-3.25,-2.75]],
    [[11.75,12.25],[-3.25,-2.75]],
    [[-12.25,-11.75],
    [-0.25,0.25]],[[-9.25,-8.75],[-0.25,0.25]],
    [[-6.25,-5.75],[-0.25,0.25]],
    [[-3.25,-2.75],[-0.25,0.25]],
    [[-0.25,0.25],[-0.25,0.25]],
    [[2.75,3.25],[-0.25,0.25]],
    [[5.75,6.25],[-0.25,0.25]],
    [[8.75,9.25],[-0.25,0.25]],
    [[11.75,12.25],[-0.25,0.25]],
    [[-12.25,-11.75],
    [2.75,3.25]],[[-9.25,-8.75],[2.75,3.25]],
    [[-6.25,-5.75],[2.75,3.25]],
    [[-3.25,-2.75],[2.75,3.25]],
    [[-0.25,0.25],[2.75,3.25]],
    [[2.75,3.25],[2.75,3.25]],
    [[5.75,6.25],[2.75,3.25]],
    [[8.75,9.25],[2.75,3.25]],
    [[11.75,12.25],[2.75,3.25]],
    [[-12.25,-11.75],
    [5.75,6.25]],[[-9.25,-8.75],[5.75,6.25]],
    [[-6.25,-5.75],[5.75,6.25]],
    [[-3.25,-2.75],[5.75,6.25]],
    [[-0.25,0.25],[5.75,6.25]],
    [[2.75,3.25],[5.75,6.25]],
    [[5.75,6.25],[5.75,6.25]],
    [[8.75,9.25],[5.75,6.25]],
    [[11.75,12.25],[5.75,6.25]],
    [[-12.25,-11.75],
    [8.75,9.25]],[[-9.25,-8.75],[8.75,9.25]],
    [[-6.25,-5.75],[8.75,9.25]],
    [[-3.25,-2.75],[8.75,9.25]],
    [[-0.25,0.25],[8.75,9.25]],
    [[2.75,3.25],[8.75,9.25]],
    [[5.75,6.25],[8.75,9.25]],
    [[8.75,9.25],[8.75,9.25]],
    [[11.75,12.25],[8.75,9.25]],
    [[-12.25,-11.75],
    [11.75,12.25]],[[-9.25,-8.75],[11.75,12.25]],
    [[-6.25,-5.75],[11.75,12.25]],
    [[-3.25,-2.75],[11.75,12.25]],
    [[-0.25,0.25],[11.75,12.25]],
    [[2.75,3.25],[11.75,12.25]],
    [[5.75,6.25],[11.75,12.25]],
    [[8.75,9.25],[11.75,12.25]],
    [[11.75,12.25],[11.75,12.25]]]

corner = corners()
total = []
for i in range(len(corner)):
    x = []
    y = []
    corner_count = 0
    xmin = corner[i][0][0]
    xmax = corner[i][0][1]
    ymin = corner[i][1][0]
    ymax = corner[i][1][1]
    for j in range(len(outputs)):
        if xmin < outputs[j][1] < xmax and ymin < outputs[j][2] < ymax:
            corner_count += 1
            x.append(outputs[j][1])
            y.append(outputs[j][2])
    total.append(corner_count)
    plt.scatter(x,y)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.show()
print(total)
