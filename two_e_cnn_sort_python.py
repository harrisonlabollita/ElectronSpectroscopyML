# CONVOLUTIONAL NEURAL NETWORK FOR TWO - ELECTRON CASE
# This network chooses whether an event was a one or two electron event.
# ------------------------------------------------------------------------------
##############################################################
#                                                            #
#  AUTHOR: HARRISON LABOLLITA                                #
#  DATE: JULY 2018                                           #
#  VERSION: 1.0                                              #
#                                                            #
##############################################################
# ------------------------------------------------------------------------------
# IMPORT NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data

filename1 = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy.csv'
grid1, outputs1 = data.get_data(filename1)
filename2 = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy2Electron.csv'
grid2, outputs2 = data.get_data(filename2)

# CREATE DATASET FROM ONE AND TWO ELECTRON DATA SIZE: 200000 EVENTS
data = []
answers = []
i = 0
while len(data) < 200000:
    num = np.random.rand()
    if num > 0.5:
        data.append(grid1[i])
        answers.append(0)
    else:
        data.append(grid2[i])
        answers.append(1)
    i +=1
data = np.array(data)/3060.

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# ORGANIZE DATASET INTO TRAINING AND TESTING DATA
num_classes = 2
x_train = data[:150000] # 150000 training events
x_train = x_train.reshape(150000,16,16,1)
y_train = np.array(answers[:150000])

x_test = data[150000:] # 50000 testing events
x_test = x_test.reshape(50000,16,16,1)
y_test = np.array(answers[150000:])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# DEFINE CNN MODEL
def Model(x_train, y_train, x_test, y_test, epochs):
    batch_size = 128
    epochs = epochs
    num_classes = 2 # 0 for one electron event, 1 for two electron event
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape= (16,16,1)))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(loss= keras.losses.categorical_crossentropy,
                    optimizer = keras.optimizers.Adadelta(),
                    metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
                batch_size= batch_size,
                epochs = epochs,
                verbose = 1, validation_data = (x_test, y_test))
    return model, history

# TRAIN MODEL/PRINT SCORE REPORT
model, history = Model(x_train, y_train, x_test, y_test, 10)


#score = model.evaluate(x_test, y_test, verbose = 0)
#print('Test loss %.2f' %(score[0]))
#print('Test accuracy %.5f' %(score[1]))


# PLOT MODEL ACCURACY/LOSS
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
