# USING MULTI OUTPUT REGRESSION FOR SEAN LIDDICK PROJECT
# ------------------------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------------------------
# Use sci-kit learns native multi-output regressor to simultaneously predict the
# x and y coordinates of the inital position of the electron for that event. The
# algorithm takes any of the native single target regressor algorithms as its
# estimator. So we use the multi layer perceptron regression, gradient boosting
# regressor and the random forest regressor as our estimators.
# ------------------------------------------------------------------------------
# IMPORT NECESSSARY LIBRARIES
import sys
import ..ProcessingData as pd
import numpy as np
import csv
from scipy.stats import chi
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------
# Define the following functions to train and test each of the aforementioned
# regressors. Then define a second function to produce and error analysis via
# a histogram of prediction error.
# ------------------------------------------------------------------------------
def multi_output_regression(train, test, grid, outputs):

    # Multi-Layer Perceptron Regressor
    input_train, input_test, output_train, actual = pd.training_testing_data(train,test ,grid, outputs)
    print('You are training on %d samples' %(len(input_train)))
    print('You are testing on %d samples' %(len(input_test)))
    multi_output_mlp = MultiOutputRegressor(MLPRegressor(solver = 'adam', learning_rate = 'adaptive',
                           max_iter = 500, early_stopping = True))
    multi_output_mlp.fit(input_train, output_train)
    prediction_mlp = multi_output_mlp.predict(input_test)
    print('Multi-Layer Perceptron')
    print(r'$R^{2}$: %.5f' %(r2_score(actual, prediction_mlp)))
    print('MSE: %.5f' %(mean_squared_error(actual, prediction_mlp)))
    print('RMSE: %.5f' %(np.sqrt(mean_squared_error(actual, prediction_mlp))))

    # Gradient Boosting Regressor
    input_train, input_test, output_train, actual = pd.training_testing_data(train,test,grid,outputs)
    print('You are training on %d samples' %(len(input_train)))
    print('You are testing on %d samples' %(len(input_test)))
    multi_output_gbr = MultiOutputRegressor(GradientBoostingRegressor(loss = 'huber'))
    multi_output_gbr.fit(input_train, output_train)
    prediction_gbr = multi_output_gbr.predict(input_test)
    print('Gradient Boosting Regressor')
    print(r'$R^{2}$: %.5f' %(r2_score(actual, prediction_gbr)))
    print('MSE: %.5f' %(mean_squared_error(actual, prediction_gbr)))
    print('RMSE: %.5f' %(np.sqrt(mean_squared_error(actual, prediction_gbr))))

    # Random Forest Regressor
    input_train, input_test, output_train, actual = pd.training_testing_data(train,test,grid,outputs)
    print('You are training on %d samples' %(len(input_train)))
    print('You are testing on %d samples' %(len(input_test)))
    multi_output_rfr = MultiOutputRegressor(RandomForestRegressor())
    multi_output_rfr.fit(input_train, output_train)
    prediction_rfr = multi_output_rfr.predict(input_test)
    print('Random Forest Regressor')
    print(r'$R^{2}$: %.5f' %(r2_score(actual, prediction_rfr)))
    print('MSE: %.5f' %(mean_squared_error(actual, prediction_rfr)))
    print('RMSE: %.5f' %(np.sqrt(mean_squared_error(actual, prediction_rfr))))

    return actual, prediction_gbr, prediction_mlp, prediction_rfr

def error_multi_output_regressor(actual, prediction_gbr, prediction_mlp, prediction_rfr):

    distance_error_gbr = []
    distance_error_mlp = []
    distance_error_rfr = []

    for i in range(len(prediction_gbr)):
        error_gbr = pd.distance_formula(actual[i,0], actual[i,1], prediction_gbr[i,0], prediction_gbr[i,1])
        distance_error_gbr.append(error_gbr)

    for i in range(len(prediction_mlp)):
        error_mlp = pd.distance_formula(actual[i,0],actual[i,1], prediction_mlp[i,0], prediction_mlp[i,1])
        distance_error_mlp.append(error_mlp)

    for i in range(len(prediction_rfr)):
        error_rfr = pd.distance_formula(actual[i,0],actual[i,1], prediction_rfr[i,0], prediction_rfr[i,1])
        distance_error_rfr.append(error_rfr)

    plt.figure(figsize = (10,7))
    n0, bins0, patches0 = plt.hist(distance_error_gbr, 10, normed =True, facecolor = 'green', edgecolor = 'black', alpha = 0.4, label ='GBR')
    n1, bins1, patches1 = plt.hist(distance_error_mlp, 10, normed = True, facecolor = 'blue', edgecolor = 'black', alpha = 0.4, label = 'MLP' )
    n2, bin2, patches2 = plt.hist(distance_error_rfr, 10, normed= True, facecolor = 'red', edgecolor = 'black', alpha = 0.4, label = 'RFR')
    y0 = chi.pdf(bins0, 2)
    y1 = chi.pdf(bins1, 2)
    y2 = chi.pdf(bin2, 2)
    plt.plot(bins0, y0, 'g--', linewidth =4)
    plt.plot(bins1, y1, 'b-.', linewidth = 4)
    plt.plot(bin2, y2, 'r:', linewidth = 4)
    plt.legend(loc ='best')
    plt.xlabel(r'distance: $\sqrt{(x_p-x)^2 + (y_p-y})^2}$')
    plt.title("Prediction Error")
    plt.show()
# ------------------------------------------------------------------------------
# MAIN CODE
# ------------------------------------------------------------------------------
# import data
filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/BetaScint2DEnergy.csv'
grid, outputs = pd.get_data(filename)
print('Finished processing data!')
# train each of the models on 250000 samples, then test on the next 1000 samples
actual, prediction_gbr, prediction_mlp, prediction_rfr = multi_output_regression(250000, 100, grid, outputs)
print('Finised testing model!')
# produce the error plots
error_multi_output_regressor(actual, prediction_gbr, prediction_mlp, prediction_rfr)
print('Finished error analysis!')
