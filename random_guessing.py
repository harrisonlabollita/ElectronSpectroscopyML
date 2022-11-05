# GENERATE RANDOM GUESS INSIDE OF EACH PIXEL
# ------------------------------------------------------------------------------
# We need this to compare our machine learning performance, so that we can
# confidentally say whether the machine is out performing a random guess.
# ------------------------------------------------------------------------------


import numpy as np
import matplotlib as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/electron_origins/src/')
import setup_electron_densities as setup
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data

filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy.csv'

grid, outputs = data.get_data(filename)

# I test the neural netowrk on 100,000 events so I only need that many. So I will
# truncate grid and outputs

random_answers = outputs[:20000]

pixels = data.find_starting_pixel(random_answers)

starting_pixels = setup.starting_pixels()
ranges = setup.ranges()


x_predictions = []
y_predictions = []
for i in range(len(pixels)):
    n = int(pixels[i])
    pixel = starting_pixels[n]
    xmin = ranges[n][0][0]
    xmax = ranges[n][0][1]
    ymin = ranges[n][1][0]
    ymax = ranges[n][1][1]
    x_predict = np.random.randint(xmin, xmax) + np.random.rand()
    y_predict = np.random.randint(ymin, ymax) + np.random.rand()
    while xmax < x_predict < xmin and ymax < ypredict < ymin:
        x_predict = np.random.randint(xmin, xmax) + np.random.rand()
        y_predict = np.random.randint(ymin, ymax) + np.random.rand()
    x_predictions.append(x_predict)
    y_predictions.append(y_predict)
print(len(x_predictions))
print(len(y_predictions))

random_errors = []
for i in range(len(random_answers)):
    error = data.distance_formula(x_predictions[i], y_predictions[i], random_answers[i][1], random_answers[i][2])
    random_errors.append(error)


plt.hist(random_errors, bins = 100)
plt.show()
