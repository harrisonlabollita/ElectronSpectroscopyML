import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/convolution_neural_network/src/')
from cnn_build import *
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score


def prob2num(start):
    highest = start[0]
    number  = 0
    for i in range(len(start)):
        if start[i] > highest:
            highest = start[i]
            number = i
    return number

def num2pixel(number):
    sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/electron_origins/src/')
    import setup_electron_densities as pix
    pixels = pix.starting_pixels()
    pixel = pixels[number]
    return pixel

filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy.csv'
grid, outputs = data.get_data(filename)
print('Done getting data!')


# TRAIN CNN
# ------------------------------------------------------------------------------
x_train, y_train, x_test, y_test, epochs, batch_size = cnn_intialize(10000,1000,30, outputs)

history, model, x_test, y_test = cnn_model(x_train,y_train,x_test,y_test, epochs, batchsize)

pixels = model.predict(x_test)

numbers = []
for i in range(len(pixels)):
    number = prob2num(pixels[i])
    numbers.append(number)

testing  = []
for i in range(len(numbers)):
    location  = num2pixel(numbers[i])
    testing.append(location)


# TRAIN MULTI - OUTPUT REGRESSOR
# ------------------------------------------------------------------------------
input_train, input_test, output_train, actual = data.training_testing_data(200000,100, grid, outputs)
multi_output_gbr = MultiOutputRegressor(GradientBoostingRegressor(loss = 'huber'))
multi_output_gbr.fit(input_train, output_train)
origin = multi_output_gbr.predict(testing)

actual = outputs[10000:11000]
x_actual = []
y_actual = []
for i in range(len(actual)):
    x_actual.append(actual[i][1])
    y_actual.append(actual[i][2])
x_actual = np.array(x_actual)
y_actual = np.array(y_actual)

actual = np.transpose(np.array([x_actual, y_actual]))

print(r2_score(actual, origin))
