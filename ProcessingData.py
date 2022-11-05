# DATA PROCESSING LIBRARY FOR PROCESSING THE SEAN LIDDICK DATA
##############################################################
#                                                            #
#  AUTHOR: HARRISON LABOLLITA                                #
#  DATE: JULY 2018                                           #
#  VERSION: 1.0                                              #
#                                                            #
##############################################################
#-----------------------------------------------------------------------------
def get_data(filename):
    import csv
    import numpy as np

    """
    Expects file "BetaScint2DEnergy.csv

    INPUTS: csv file of Sean Liddick data

    OUTPUTS: The inputs for the machine learning algorithm, a grid of the detector or a vector depending on the
    need, adjust accordingly. And a vector of outputs containing the initial energy and origin of electron
    """

    file = open(filename)
    events = csv.reader(file)
    data = []
    for event in events:
        data.append(event)
    data = np.asarray(data)
    N, M = data.shape
    val = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            val[i][j] = float(data[i][j])
    grid = val[:,0:256]
    outs = val[:,256:]
    events = []
    outputs = []
    for i in range(len(grid)):
        events.append(np.transpose(grid[i,:].reshape((16,16)))) #We need to flip the matrix to match the correct spatial orientation
    for i in range(len(outs)):
        outputs.append(outs[i,:])
    return events, outputs



# -----------------------------------------------------------------------------
def get_data2(filename):
    import csv
    import numpy as np

    """
    Expects file "BetaScint2DEnergy.csv

    INPUTS: csv file of Sean Liddick data

    OUTPUTS: The inputs for the machine learning algorithm, a grid of the detector or a vector depending on the
    need, adjust accordingly. And a vector of outputs containing the initial energy and origin of electron
    """

    file = open(filename)
    events = csv.reader(file)
    data = []
    for event in events:
        data.append(event)
    data = np.asarray(data)
    N, M = data.shape
    val = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            val[i][j] = float(data[i][j])
    grid = val[:,0:256]
    outs = val[:,256:]
    events = []
    outputs = []
    for i in range(len(grid)):
        events.append(np.transpose(grid[i,:].reshape((16,16)))) #We need to flip the matrix to match the correct spatial orientation
    for i in range(len(outs)):
        outputs.append(outs[i,:])
    return events, outputs


def one_or_two_electrons(data):
    """This function determines whether the event i was a one or two electron event"""
    one_electron = []
    two_electron = []
    for i in range(len(data)):
        Energy_1 = data[i][0]
        x_1 = data[i][1]
        y_1 = data[i][2]
        Energy_2 = data[i][3]
        x_2 = data[i][4]
        y_2 = data[i][5]

    if Energy_1 == 0 or x_1 == 0 or y_1 == 0:
        one_electron.append(data[i])
    elif Energy_2 == 0 or x_2 == 0 or y_2 == 0:
        one_electron.append(data[i])
    else:
        two_electron.append(data[i])
    return one_electron, two_electron
# -----------------------------------------------------------------------------
def training_testing_data(N, M, grid, outputs):

    """
    This functions creates all of the training and testing
    sets for the machine learning algorithms
    INPUTS:     N = size of training set
                M = size of testing set
                grid = all of the detector grids, code then finds all of the pixels with the highest energy in each grid
                this information is then saved in a list of locations
                outputs = vector of all of the x and y positions of the starting electron for each event

    OUTPUTS:    input_train = set of training data (inputs)
                output_train - set of training data (outputs)
                input_test = set of testing data (inputs)
                actual = the correct answers to the input_test set. This is used for comparison to the predictions
                by the machine learning algorithms

    """
    import numpy as np

    initial_position_x = []
    initial_position_y = []
    locations = []
    for i in range(len(grid)):
        energy, location = max_energy_matrix(grid[i])
        locations.append(location)
        initial_position_x.append(outputs[i][1])
        initial_position_y.append(outputs[i][2])
    x_pixel = []
    y_pixel = []
    for i in range(len(locations)):
        x_pixel.append(locations[i][1])
        y_pixel.append(locations[i][0])

    x_input_train = np.array(x_pixel[:N])
    y_input_train = np.array(y_pixel[:N])

    input_train = np.transpose(np.array([x_input_train, y_input_train]))


    x_input_test = np.array(x_pixel[N:N+M])
    y_input_test = np.array(y_pixel[N:N+M])

    input_test = np.transpose(np.array([x_input_test, y_input_test]))


    x_output_train = np.array(initial_position_x[:N])
    y_output_train = np.array(initial_position_y[:N])

    output_train = np.transpose(np.array([x_output_train, y_output_train]))


    x_actual = np.array(initial_position_x[N:N+M])
    y_actual = np.array(initial_position_y[N:N+M])

    actual = np.transpose(np.array([x_actual, y_actual]))
    return input_train, input_test, output_train, actual

def max_energy_matrix(grid):
    """
    This function finds the maximum energy deposited in a single pixel i,j

    INPUT: Matrix representing the detector grid.

    OUTPUT: Maximum energy deposited in single pixel on the grid.
            Format: pixel = [i, j], where i = row and j = column

    """

    N, M = grid.shape
    max_energy = grid[0][0]
    pixel = [0,0]
    for i in range(N):
        for j in range(M):
            if max_energy > grid[i][j]:
                max_energy = max_energy
                pixel = pixel
            else:
                max_energy = grid[i][j]
                pixel = [i,j]
    return max_energy, pixel

def distance_formula(x, y, xhat, yhat):

    """
    Computes the distance between the points (x, y) and (xhat, yhat)

    INPUTS: (x, y) and (xhat, yhat)
    OUTPUT: ditance between these two points

    """
    import numpy as np
    return (np.sqrt( (yhat - y)**2 + (xhat - x)**2))

def grid2img(grids):

    """
    This function converts each grid into a picture that will be used as
    data for the convolutonal neural network.

    INPUTS: detector grid i
    OUTPUTS: saved image file for the convolutional neural network to use.
            The images are saved in folder 'CNN_DATA'
    """

    import matplotlib.pyplot as plt
    # I have limited to just the first 1000, but this can be changed.
    for i in range(len(grids)-700000):
        plt.figure(figsize = (1,1))
        fig = plt.imshow(grids[i], cmap = 'binary', origin = 'lower')
        fig.axes.get_xaxis().set_ticks([])
        fig.axes.get_yaxis().set_ticks([])
        fig.axes.get_xaxis().set_ticklabels([])
        fig.axes.get_yaxis().set_ticklabels([])
        plt.savefig("/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/CNN_DATA/grid_%d.png" %(i))
        plt.close()

def nonzero_pixels(grid):
    pixels = []
    grid = grid.reshape(256)
    for i in range(len(grid)):
        if grid[i] > 0:
            pixels.append((grid[i],i))
    return pixels


def find_starting_pixel(outputs):
    """
    This function finds the answers for the convolutional neural network.
    It finds the correct pixel that the electron started in. By taking the x and y
    coordinates of each electron and comparing it to a list of all the possible
    electron origins. When it finds a match it takes the number pixel that it
    started in. See file pixel_locations.txt for more information.

    INPUT: outputs = vector of all of the electron origins.
    OUTPUT: pixels = list of the pixel that the electron started in for its corresponding event

    """

    import sys
    sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/python/electron_origins/src/')
    import setup_electron_densities as setup
    acceptable_ranges = setup.ranges()
    pixels = []
    for i in range(len(outputs)):
        for j in range(len(acceptable_ranges)):
            xmin = acceptable_ranges[j][0][0]
            xmax = acceptable_ranges[j][0][1]
            ymin = acceptable_ranges[j][1][0]
            ymax = acceptable_ranges[j][1][1]
            if xmin < outputs[i][1] < xmax and ymin < outputs[i][2] < ymax:
                pixels.append(j)
    return pixels

def cnn_train_test_data(train_size, test_size, outputs):

    """
    This function creates the training and testind data used in the convolutonal neural network.
    NOTE: user needs to be aware of the amount of avaliable data. If more pictures need to be generated
          see file cnn_data_setup.py

    INPUTS: train_size = size of training set
            test_size = size of testing set
            outputs = needs all of the outputs so that it can find the starting pixels of the electrons.
    OUTPUTS: x_train = training set for convolutional neural network (inputs)
                       --> the images of the detector
             y_train = training set for convolutional neural network (output)
                       --> the number pixel that the electron started in
             x_test =  testing set for the convolutional neural network (inputs)
             y_test =  testing set for the convolutional neural network (outputs)

             The testing sets are used as validation when training the main convolutonal neural network
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # FIND THE PIXEL THE ELECTRON STARTED IN
    answers = find_starting_pixel(outputs)
    data = []
    length = train_size + test_size
    for i in range(length):
        filename =  "/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/CNN_DATA/grid_%d.png" %(i)
        img = plt.imread(filename)
        img = img[:,:,1]
        data.append(img)
    y_train = np.array(answers[:train_size])
    y_test = np.array(answers[train_size:train_size+test_size])
    x_train = np.array(data[:train_size])
    x_test = np.array(data[train_size:train_size +test_size])

    return x_train, y_train, x_test, y_test


def print_stats(pixels, count_in, count_total):

    """
    This function is used to print the statistics from electron_densities.py.
    INPUTS: pixels = list of all of the starting pixels
            count_in = array of all of the electrons that fell inside its appropriate pixel
            count_total = array of all the times that this pixel was the pixel with the highest energy
    OUTPUTS: the statistics for each pixel.
                                 EXAMPLE
    Pixel                 Inside                Total                   P
    ----------------------------------------------------------------------
    [3,3]                  6000                 8000                   0.75
    and so on for all 100 of the starting pixels.

    """

    import numpy as np
    probabilites = []
    print("Pixel \t\t Inside \t Total \t\t P")
    print("-------------------------------------------------------------")
    for i in range(len(pixels)):
        p = count_in[i]/count_total[i]
        probabilites.append(p)
        print("%d,%d \t\t %d \t\t %d \t\t %.3f" %(pixels[i][0], pixels[i][1], count_in[i], count_total[i], p))
    print("--------------------------------------------------------------")
    print("Average: \t %d \t\t %d \t\t %.3f" %(np.floor(np.mean(count_in)), np.floor(np.mean(count_total)), np.mean(probabilites)))
