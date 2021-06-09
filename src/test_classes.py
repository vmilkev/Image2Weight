import farmdata as fd
import bwmodel as bwm

import matplotlib.pyplot as plt
import numpy as np
import random
import math
#from scipy import stats

readFile = '../data/2020-06-12_Standard_Ejby_Jersey.csv'
#readFile = '../data/2020-06-11_Standard_Nyborg_Jersey.csv'
#readFile = '../data/2020-06-12_Standard_Glamsbjerg.csv'
#readFile = '../data/2020-06-15_Standard_Skjern.csv'
delim = ';'

# Instantiating an FarmData class object
d = fd.FarmData(readFile, delim)

# See the data header
#d.prt_fheader()

# Use the header to create a positions list
# of records which need to be extracted
#           t id  w  mesures
pos = list([0, 1, 2, 6])

# Extract data into object's variables
idata = d.extr_fdata(pos)

# Print part of extracted data, nr. of records
d.prt_idata(idata, 0)

# how to get the individual data
# here 10th record from the end
id = d.u_id[-10]
ind1 = idata[id]

# remove outliers:
#
# 'zss' is sample based Z-score method
# 'zss2' is sample based modified Z-score method
# 'cls' is sample based clustering
# 'zs/cls' is the combination of both mentioned methods
#
#idata2 = d.rem_outlier( idata, 'zs/cls', 2, 1.5 )
#idata2 = d.rem_outlier( idata, 'cls', 5, 0 )
#idata2 = d.rem_outlier( idata, 'zss', 2.5, 0 )
idata2 = d.rem_outlier( idata, 'zss2', 3.0, 0 )
#idata2 = d.rem_outlier( idata2, 'zss', 2.0, 0 )

print(f'\tlen idata with outliers: {len(idata)}')
print(f'\tlen idata without outliers: {len(idata2)}')

checkoutliers_all = False
checkoutliers_ind = False

if checkoutliers_all:
    # collect all observed body weights over time
    # and plot them to see possible data clustering
    # and the amount of otliers
    #
    # 1) collect data before outlier removal
    all_weights = []
    all_times = []
    for i in d.u_id:
        tmp = idata[i]
        tmp2 = tmp[:,1]
        tmp3 = tmp[:,0]
        all_weights = all_weights + tmp2.tolist()
        all_times = all_times + tmp3.tolist()

    plt.scatter( all_times[0:], all_weights[0:] )

    # 2) collect data after outlier removal
    all_weights = []
    all_times = []
    for i in d.u_id:
        tmp = idata2[i]
        tmp2 = tmp[:,1]
        tmp3 = tmp[:,0]
        all_weights = all_weights + tmp2.tolist()
        all_times = all_times + tmp3.tolist()

    plt.scatter( all_times[0:], all_weights[0:] )

    # 3) show all data (in diff. colors) in the same plot window
    plt.show()

    if checkoutliers_ind:
        # check for every individual
        # checking a quality of outlier detection
        for i in d.u_id:
            tmp0 = idata[i]
            tmp02 = tmp0[:,1]
            tmp03 = tmp0[:,0]
            tmp2 = idata2[i]
            tmp22 = tmp2[:,1]
            tmp23 = tmp2[:,0]
            plt.scatter( tmp03[0:], tmp02[0:] )
            plt.scatter( tmp23[0:], tmp22[0:] )
            plt.show()

# impute missing feature (conur data)
d.impute(idata2)

# print out some data to files
dataprint = False
if dataprint:
    sz1 = 80 # training dataset, %
    sz2 = 20 # validation dataset, %
    f1 = open("training.dat", "a")
    f2 = open("validation.dat", "a")
    for i in range(1, int( len(d.u_id) )):
        num = random.randint(1, 100)
        if num > sz2:
            np.savetxt(f1, idata2[ d.u_id[i] ][:,1:] )
            f1.write("\n")
        else:
            np.savetxt(f2, idata2[ d.u_id[i] ][:,1:] )
            f2.write("\n")
    f1.close()
    f2.close()

# check imputed data
checkimpute = False

if checkimpute:
    num = 4
    for i_range in range(0,len( idata2[ d.u_id[num] ][:,0] )):
        plt.plot( idata2[ d.u_id[num] ][i_range,2:], 'bo' )
        plt.show()


# Creating models using the class BWModel:
#
# prepare training and validation data sets: target and features
sz1 = 80 # training dataset, %
sz2 = 20 # validation dataset, %
trWeights = []
trFeatures = []
vlWeights = []
vlFeatures = []
for i in range(1, int( len(d.u_id) )):
    num = random.randint(1, 100)
    timeDataLength = len(idata2[ d.u_id[i] ][:,1])
    if timeDataLength >= 10:
        range = math.floor(0.8 * timeDataLength)
        if num > sz2:
            #print(f'\tlenth of idata2[ d.u_id[i] ][:,1] = {len(idata2[ d.u_id[i] ][:,1])}')
            tmp2 = idata2[ d.u_id[i] ][0:range-1,1]
            tmp3 = idata2[ d.u_id[i] ][0:range-1,2:]
            trWeights = trWeights + tmp2.tolist()
            trFeatures = trFeatures + tmp3.tolist()
        else:
            tmp2 = idata2[ d.u_id[i] ][range:,1]
            tmp3 = idata2[ d.u_id[i] ][range:,2:]
            vlWeights = vlWeights + tmp2.tolist()
            vlFeatures = vlFeatures + tmp3.tolist()

# free memory
d.clear()

# create an object of the class
m1 = bwm.BWModel(trWeights, trFeatures, vlFeatures, 'LR')

m1.fit()
pdWeights = m1.pred()

print(np.corrcoef(vlWeights, pdWeights))

plt.scatter( vlWeights, pdWeights )
plt.show()



 

