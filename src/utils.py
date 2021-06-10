import random
import math
import numpy as np
import matplotlib.pyplot as plt

class StudyData:
    # collection of methods to produce Training and Validation Data sets

    def __init__(self, id_list, data, trainingpart, method):
        # class constructor

        self.data = data
        self.u_id = id_list
        self.meth = method
        self.trpart = trainingpart # % of the data used for training; trainingpart = (0,100)

    def get(self):
        if self.meth == 'ts':
            return self.__ts()
        else:
            return
    
    def __ts(self):
        # prepare training and validation data sets: target and features

        sz1 = self.trpart        # training dataset, %
        sz2 = 100 - self.trpart  # validation dataset, %
        tsData = 0.8             # part of time series data used fro training
        trTarget = []
        trFeatures = []
        vlTarget = []
        vlFeatures = []
        for i in range(1, int( len(self.u_id) )):
            num = random.randint(1, 100)
            timeDataLength = len(self.data[ self.u_id[i] ][:,1])
            if timeDataLength >= 10:
                trrange = math.floor(tsData * timeDataLength)
                if num > sz2:
                    #print(f'\tlenth of self.data[ self.u_id[i] ][:,1] = {len(self.data[ self.u_id[i] ][:,1])}')
                    tmp2 = self.data[ self.u_id[i] ][0:trrange-1,1]
                    tmp3 = self.data[ self.u_id[i] ][0:trrange-1,2:]
                    trTarget = trTarget + tmp2.tolist()
                    trFeatures = trFeatures + tmp3.tolist()
                else:
                    tmp2 = self.data[ self.u_id[i] ][trrange:,1]
                    tmp3 = self.data[ self.u_id[i] ][trrange:,2:]
                    vlTarget = vlTarget + tmp2.tolist()
                    vlFeatures = vlFeatures + tmp3.tolist()
        
        return trTarget, trFeatures, vlTarget, vlFeatures;


class IOData:
    # some in/out interfaces

    def fwrite(self, data, id_list, trainingpart, trainingDataFile, validationDataFile):
                                     # trainingpart is a training dataset, % of the data used for training; trainingpart = (0,100)
        sz2 = 100 - trainingpart     # validation dataset, %
        f1 = open(trainingDataFile, "a")
        f2 = open(validationDataFile, "a")
        for i in range(1, int( len(id_list) )):
            num = random.randint(1, 100)
            if num > sz2:
                np.savetxt(f1, data[ id_list[i] ][:,1:] )
                f1.write("\n")
            else:
                np.savetxt(f2, data[ id_list[i] ][:,1:] )
                f2.write("\n")
        f1.close()
        f2.close()

    def compare_outlier(self, id_list, rowData, clearData, interactive):
        # collect all observed body weights over time
        # and plot them to see possible data clustering
        # and the amount of otliers
        #
        # 1) collect data before outlier removal
        all_weights = []
        all_times = []
        for i in id_list:
            tmp = rowData[i]
            tmp2 = tmp[:,1]
            tmp3 = tmp[:,0]
            all_weights = all_weights + tmp2.tolist()
            all_times = all_times + tmp3.tolist()

        plt.scatter( all_times[0:], all_weights[0:] )

        # 2) collect data after outlier removal
        all_weights = []
        all_times = []
        for i in id_list:
            tmp = clearData[i]
            tmp2 = tmp[:,1]
            tmp3 = tmp[:,0]
            all_weights = all_weights + tmp2.tolist()
            all_times = all_times + tmp3.tolist()

        plt.scatter( all_times[0:], all_weights[0:] )

        # 3) show all data (in diff. colors) in the same plot window
        plt.show()

        if interactive:
            # check for every individual
            # checking a quality of outlier detection
            for i in id_list:
                tmp0 = rowData[i]
                tmp02 = tmp0[:,1]
                tmp03 = tmp0[:,0]
                tmp2 = clearData[i]
                tmp22 = tmp2[:,1]
                tmp23 = tmp2[:,0]
                plt.scatter( tmp03[0:], tmp02[0:] )
                plt.scatter( tmp23[0:], tmp22[0:] )
                plt.show()

