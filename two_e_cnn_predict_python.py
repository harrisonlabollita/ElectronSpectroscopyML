# CONVOLUTIONAL NEURAL NETWORK FOR PREDICITING ORIGINS OF ELECTRONS
# This network will predict the origins of the electron in the two electron case
# in a similar fashion to the one-electron neural network.
##############################################################
#                                                            #
#  AUTHOR: HARRISON LABOLLITA                                #
#  DATE: JULY 2018                                           #
#  VERSION: 1.0                                              #
#                                                            #
##############################################################

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/convolution_neural_network/src/')
import sort_easy_hard_init as sorter
filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy2Electron.csv'
grid, outputs = data.get_data(filename)
grid = np.array(grid)/3060. #Normalize the data
print('Load data: DONE')


data = []
answers = []
for k in range(len(grid)):
    test = grid[k]
    N, M = test.shape
    checker = np.zeros((N,M))
    v = []
    for i in range(N):
        for j in range(M):
            if sorter.excluded(checker, i, j) == True:
                continue
            count = 0
            if test[i][j] > 0:
                checker, count = sorter.neighbors(test,checker,i,j,count)
                if count == 0:
                    count+=1
                v.append(count)
    if len(v) == 2:
        data.append(grid[k])
        answers.append(outputs[k])

x_train = np.array(data[:len(data)-10000])
x_train = x_train.reshape(len(x_train), 16, 16, 1)
x_test = np.array(data[len(data)-10000:])
x_test = x_test.reshape(len(x_test), 16, 16, 1)

out = []
for answer in answers:
    out.append([answer[1], answer[2], answer[4], answer[5]])
y_train = np.array(out[:len(data)-10000])
y_test = np.array(out[len(data)-10000:])

import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def Model(x_train, y_train, x_test, y_test, epochs):
    batch_size = 128
    epochs = epochs
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape= (16,16,1)))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, input_dim = 100, activation = 'relu'))
    model.add(Dense(4, activation='linear'))
    model.compile(loss= 'mse', optimizer='adam', metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
            batch_size= batch_size,
            epochs = epochs,
            verbose = 1, validation_data = (x_test, y_test))

    model.summary()

    return model, history


model, history = Model(x_train, y_train, x_test, y_test,100)

plt.style.use('seaborn')
fig, ax = plt.subplots(1,2, figsize = (7,5))
ax[0].set_aspect('auto')
ax[0].plot(history.history['acc'], label = 'Train')
ax[0].plot(history.history['val_acc'],label = 'Test')
ax[0].set_title('Multi-Event Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc = 'best')
ax[1].set_aspect('auto')
ax[1].plot(history.history['loss'], label = 'Train')
ax[1].plot(history.history['val_loss'],label = 'Test')
ax[1].set_title('Multi-Event Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend(loc = 'best')
fig.subplots_adjust(right = 2)
#plt.savefig('Multi-Event Model.png', bbox_inches = 'tight')
plt.show()
