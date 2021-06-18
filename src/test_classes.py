import farmdata as fd
#import bwmodel as bwm
from bwmodel import BWModel
from utils import IOData, StudyData

import matplotlib.pyplot as plt
import numpy as np

#readFile = '../../data/2020-06-12_Standard_Ejby_Jersey.csv'
#readFile = '../../data/2020-06-11_Standard_Nyborg_Jersey.csv'
#readFile = '../../data/2020-06-12_Standard_Glamsbjerg.csv'
readFile = '../../data/2020-06-15_Standard_Skjern.csv'
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
# check the quality of outlier detection
if checkoutliers_all:
    prdat = IOData()
    prdat.compare_outlier(d.u_id, idata, idata2, checkoutliers_ind)
    del prdat

# impute missing feature (conur data)
d.impute(idata2)

# print out some data to files
dataprint = False
if dataprint:
    prdat = IOData()
    prdat.fwrite(idata2, d.u_id, 80, "training.dat", "validation.dat")
    del prdat

# check imputed data
checkimpute = False
if checkimpute:
    num = 4
    for i_range in range(0,len( idata2[ d.u_id[num] ][:,0] )):
        plt.plot( idata2[ d.u_id[num] ][i_range,2:], 'bo' )
        plt.show()


# prepare training and validation data sets: target and features
#sdat = StudyData(d.u_id, idata2, 80, 'ts')
#sdat = StudyData(d.u_id, idata2, 80, 'ra')
sdat = StudyData(d.u_id, idata2, 80, 'di')

trWeights, trFeatures, vlWeights, vlFeatures = sdat.get()

# print(trWeights)
# print(type(trWeights))
# print(len(trWeights))
# #print(len(trWeights[0]))
# print(trFeatures)

# raise SystemExit

# free memory
d.clear()
del d

# Creating models using the class BWModel:

# 1. 'lr': linear regression
# 2. 'rr': ridge regression, alpha = 0.5 (default)
# 3. 'la': lasso regression, alpha = 0.5 (default)
# 4. 'dt': decission tree regressor, random_state = 0 (default)
# 5. 'rf': random forest regressor, random_state = 0 (default)
# 6. 'ab': ada boost regressor, random_state = 0 (default)
# 7. 'sv': support vector regressor, C = 10 (default)

methods = ['lr', 'rr', 'la', 'dt', 'rf', 'ab', 'sv']
fig, axs = plt.subplots( len(methods) )
corr = []

for i in range(0, len(methods)):
    
    print(methods[i])

    m = BWModel()
    m.fit(trWeights, trFeatures, methods[i])
    pdWeights = m.pred( vlFeatures )

    # other mode:
    # ftmodel = m.fit(trWeights, trFeatures, methods[i])
    # pdWeights = ftmodel.predict( vlFeatures )

    a = np.corrcoef(vlWeights, pdWeights)
    corr.append(a[0,1])
    axs[i].scatter( vlWeights, pdWeights )
    #axs[i].title.set_text( methods[i] + ", corr = " + str(a[0,1]) )
    del m

print(corr)
#plt.tight_layout()
plt.show()
