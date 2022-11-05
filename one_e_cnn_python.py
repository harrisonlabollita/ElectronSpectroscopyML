# CONVOLUTIONAL NEURAL NETWORK FOR ONE - ELECTRON CASE
# Inspired by the work of Nicholas, this is a python version of
# of the neural network using Keras.
# ------------------------------------------------------------------------------
##############################################################
#                                                            #
#  AUTHOR: HARRISON LABOLLITA                                #
#  DATE: JULY 2018                                           #
#  VERSION: 1.0                                              #
#                                                            #
##############################################################
# ------------------------------------------------------------------------------
#  IMPORT NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/electron_origins/src/')
import setup_electron_densities as setup

# IMPORT DATA
filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy.csv'
grid, outputs = data.get_data(filename)
grid = np.array(grid)/3060. # Normalize the energy depositions

# MAKE TRAINING AND TESTING SETS

x_train = np.array(grid[:333333])
x_train = x_train.reshape(333333,16,16,1)

x_test = np.array(grid[500000:520000])
x_test = x_test.reshape(20000,16,16,1)

answers=[]
for i in range(len(outputs)):
    answers.append([outputs[i][1], outputs[i][2]])
answers = np.array(answers)

y_train = answers[:333333]
y_test = answers[500000:520000]

# we now have two mutually exclusive data sets ready for the convolutional neural
# network

# BUILD NETWORK
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def Model(x_train, y_train, x_test, y_test, epochs):
    batch_size = 128
    epochs = epochs
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape= (16,16,1)))
    model.add(Flatten())
    model.add(Dense(512, input_dim = 100, activation = 'relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss= 'mse', optimizer='adam', metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
            batch_size= batch_size,
            epochs = epochs,
            verbose = 1, validation_data = (x_test, y_test))

    model.summary()

    return model, history

# TRAIN NETWORK
model, history = Model(x_train, y_train, x_test, y_test, 10)


# PLOT MODEL ACCURACY/LOSS
plt.style.use('seaborn-poster')
fig, ax = plt.subplots(1,2)
ax[0].plot(history.history['acc'], label = 'Train')
ax[0].plot(history.history['val_acc'], label = 'Test')
ax[0].set_title('Single Electron Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc = 'best')
ax[1].plot(history.history['loss'], label = 'Train')
ax[1].plot(history.history['val_loss'], label = 'Test')
ax[1].set_title('Single Electron Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend(loc = 'best')
fig.subplots_adjust(right = 2)
plt.show()

# PREDICT
prediction = model.predict(x_test)

# MODEL ERRORS
errors = []
for i in range(len(prediction)):
    error = data.distance_formula(prediction[i][0], prediction[i][1], y_test[i][0], y_test[i][1])
    errors.append(error)

# RANDOM GUESS
random_answers = outputs[:20000]

starting_pixels = setup.starting_pixels()
ranges = setup.ranges()

pixels = data.find_starting_pixel(random_answers)
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

num = np.mean(random_errors)
count = 0
for error in errors:
    if error <= num:
        count +=1
print(count)
print(count/len(errors))


plt.style.use('default')
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
fig, ax = plt.subplots()
axins = zoomed_inset_axes(ax, 2.5, loc=3)
ax.imshow(grid[500001], cmap='BuGn', origin = 'lower')
ax.scatter((prediction[1][0])/3 + 8, (prediction[1][1])/3 +8, c = 'blue', label = 'ConvNet')
ax.scatter((y_test[1][0])/3 + 8, (y_test[1][1])/3 + 8, c = 'r', alpha =0.9, label = 'Actual')
axins.imshow(grid[500001], cmap='BuGn', origin = 'lower')
axins.scatter((prediction[1][0])/3 + 8, (prediction[1][1])/3 +8, c = 'blue', label = 'ConvNet')
axins.scatter((y_test[1][0])/3 + 8, (y_test[1][1])/3 + 8, c = 'r', alpha =0.9, label = 'Actual')
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.legend(loc = 'best')
x1, x2, y1, y2 = 4,10,10,14
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
axins.axes.get_xaxis().set_visible(False)
axins.axes.get_yaxis().set_visible(False)
#plt.savefig('ConvNet_prediction.png')
plt.show()

# PLOT MACHINE LEARNING ERROR VS RANDOM GUESSING ERROR
plt.style.use('seaborn-poster')
fig, (ax1,ax3) = plt.subplots(2, 1)
ax1.set_title('Single-Electron Model vs. Random Guess')
ax1.hist(errors, bins = 100, histtype = 'step', normed = True, linewidth = 3, label = 'ConvNet')
ax1.hist(random_errors, bins = 100, histtype = 'step', normed = True, linewidth = 3, label = 'Random Guess')
ax1.legend(loc = 'best', fontsize = 10)
ax1.set_xlabel('Distance (mm)')

ax3.set_title('Log Scale: Single-Electron Model vs. Random Guess')
ax3.hist(errors, bins = 100, histtype = 'step', log =True, normed = True, linewidth = 3,label = 'ConvNet')
ax3.hist(random_errors, bins = 100, histtype = 'step', log = True, normed = True, linewidth = 3, label = 'Random Guess')
ax3.legend(loc = 'best', fontsize = 10)
ax3.set_xlabel('Distance (mm)')
plt.subplots_adjust(hspace = 0.5)
#plt.savefig('Norm and Log Single-Electron Model vs Random Guess.png')
plt.show()

plt.style.use('seaborn-poster')
fig, ax = plt.subplots(figsize = (7,5))
ax.set_title('Zoom: Single - Electron Model vs. Random Guess')
ax.hist(errors, bins = 100, histtype = 'step', normed = True, linewidth = 3, label = 'ConvNet')
ax.hist(random_errors, bins = 100, histtype = 'step', normed = True, linewidth = 3, label = 'Random Guess')
ax.set_xlim([0,3])
ax.legend(fontsize = 12)
ax.set_xlabel('Distance (mm)')
#plt.savefig('Zoom: Single-Electron Model vs Random Guess.png',bbox_inches = 'tight')
plt.show()
