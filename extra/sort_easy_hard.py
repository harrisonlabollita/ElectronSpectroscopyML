import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/convolution_neural_network/src/')
import sort_easy_hard_init as sorter
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/python/')
import ProcessingData as data

filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Research Experience for Undegraduates/MSU REU/MSU REU Project/BetaScint2DEnergy2Electron.csv'
grid, outputs = data.get_data(filename)

hard = [] # hard cases will be when len(v) != 2
for i in range(len(grid)):
    test = grid[i]
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
                if count ==0:
                    count+=1
                v.append(count)
    if len(v) != 2:
        hard.append(grid[i])

# We found that 15 % of the cases are hard. So the CNN should perform very well.
print(len(hard)/len(grid))
