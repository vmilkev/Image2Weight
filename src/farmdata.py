import csv
import datetime
import math
import numpy as np
import gc

class FarmData:
    
    # MEMBERS:

    # self.srcfile   - name of source file
    # self.dlmt      - delimeter type
    # self.time      - array of time stamps
    # self.id        - array of IDs
    # self.weight    - array of reference weights
    # self.mes_data  - array of measured conturs' points
    # self.nzcolumns - numbe rof columns in data file
    # self.unique_id - map of unique IDs and their related indexes in data arrays
    # self.u_id      - list of unique IDs

    # PUBLIC METHODS:

    # prt_idata(self, idata, nr_lines)                            - print-out some processed data
    # prt_fheader(self)                                           - method to be used in order to check a header (structure) of .csv file
    # extr_fdata(self, positions)                                 - read data from .csv file to the class member variables
    # clear( self )                                               - free memory
    # rem_outlier( self, data, method, threshold1, threshold2 )   - entry point for different outlier removal methods
    # impute( self, data )                                        - impute data

    # PRIVATE METHODS:

    # __init__(self, source_file, source_delimeter)               - class constructor
    # __dbg_prt_fcontent(self, nr_lines, what)                    - print-out some red data to check (unprocessed data)
    # __get_columns(self)                                         - get number of columns in .csv file
    # __pars_weight( self,str_weight )                            - parsing weight records which apiars like [num]
    # __pars_time( self, t )                                      - parsing day/time and convert to int
    # __count_nzrecords(self, pos_records)                        - count the number of non-zero records
    # __unique(self)                                              - get list of indexes for each unique ID in original data set
    # __get_idata(self, id)                                       - form a data (record) matrix for particular individual
    # __clear( self )                                             - free memory
    # __zscorecluster_s( self, data, t_col, w_col, w_var, w_std ) - Mixed Clustering / Z-score method of outlier detection
    # __cluster_s( self, data, t_col, w_col, w_var, w_mean )      - Clustering method of outlier detection
    # __zscore_s( self, data, icol, threshold )                   - sample based Z-score method of outlier detection
    # __zscore2_s( self, data, icol, threshold )                  - sample based modified Z-score method of outlier detection


    def __init__(self, source_file, source_delimeter):
        # class constructor
        self.srcfile = source_file
        self.dlmt = source_delimeter
        
    def __dbg_prt_fcontent(self, nr_lines, what):
        # ! do not use it in production - only for debuging
        #
        # print-out some red data to check (unprocessed data)
        # nr_lines - number of lines to print
        # what     - if what == all print entire row
        #            if what == weight print time id and weight
        #            if what == mesr print contur data
        print(self.srcfile)
        for i in range(nr_lines):
            if what == "all":
                print(f'\tTime: {self.time[i, 0]} id: {self.id[i, 0]} weight: {self.weight[i, 0]} mesr: {self.mes_data[i,0:-1]}')
            if what == "weight":
                print(f'\tTime: {self.time[i, 0]} id: {self.id[i, 0]} weight: {self.weight[i, 0]}')
            if what == "mesr":
                print(f'\tmesr: {self.mes_data[i,0:-1]}')

    def prt_idata(self, idata, nr_lines):
        # print-out some processed data
        # nr_lines - number of lines to print
        print(f'\tprocessed file name: {self.srcfile}')
        l = 0
        for i in self.u_id:
            if l < nr_lines:
                data = idata[i]
                #print(f'\tdata shape {data.shape}')
                print(f'\tID: {i}; time: {data[1,0]}; weight: {data[1,1]}; conturs: {data[1,2:]}')
                l = l + 1
            else:
                break
                
    def prt_fheader(self):
        # method to be used in order to check a header (structure) of .csv file
        with open(self.srcfile) as csv_file:
            # read file for a first time
            # in order to get a number of rows and cols
            csv_reader = csv.reader( csv_file, delimiter = self.dlmt )
            lines = 0
            for row in csv_reader:
                if lines == 0:
                    # print header to see the structure of the file
                    print('The csv file: ', self.srcfile)
                    print('The header:')
                    print(', '.join(row))
                    print(f'\tNumber of columns: {len(row)}')
                    return

    def __get_columns(self):
        # get number of columns in .csv file
        with open(self.srcfile) as csv_file:
            # read file for a first time
            # in order to get a number of cols
            csv_reader = csv.reader( csv_file, delimiter = self.dlmt )
            lines = 0
            for row in csv_reader:
                if lines == 0:
                    self.nzcolumns = len(row)
                    return

    def __pars_weight( self,str_weight ):
        # privat method for parsing weight records which apiars like [num]
        str_weight = str_weight.replace(",", ".")
        x = str_weight.split("_")
        if len(x) > 1:
            x = x[-1].split("]")
        else:
            x = str_weight.split("[")
            x = x[-1].split("]")
        return float(x[0])

    def __pars_time( self, t ):
        # privat method for parsing day/time and convert to int
        # timestamp is the number of seconds between a particular date and January 1, 1970 at UTC
        date_time = datetime.datetime.strptime(t, '%d-%m-%Y %H:%M:%S.%f ')
        a_timedelta = date_time - datetime.datetime(1970, 1, 1)
        seconds = a_timedelta.total_seconds()
        return int( math.ceil(seconds) ) # seconds
    
    def __count_nzrecords(self, pos_records):
        # count the number of non-zero records
        with open(self.srcfile) as csv_file:
            # read file for a first time
            # in order to get a number of rows and cols
            csv_reader = csv.reader( csv_file, delimiter = self.dlmt )
            lines = 0
            for row in csv_reader:
                if lines == 0:
                    lines += 1
                else:
                    # we shall read only animals with records
                    if self.__pars_weight( row[pos_records] ) != 0.0:    
                        lines += 1
            return lines-1

    def __unique(self):
        # function to get list of indexes for each
        # unique ID in original data set              
        # (1) find unique IDs
        self.u_id = []
        for x in self.id[:,0]:
            if x not in self.u_id:
                self.u_id.append(x)                
        # (2) define the map: for each unique id there is
        #     the list of indexes in the data arrays
        #     such as [id][list of idexes in data]
        self.unique_id = {}        
        dict_size = 0
        for x1 in self.u_id:
            index = 0
            unique_list = []
            for x2 in self.id[:,0]:
                if x2 == x1:
                    unique_list.append(index)
                index += 1
            self.unique_id[x1] = unique_list
            dict_size += 1
    
    def extr_fdata(self, positions):
        # read data from .csv file to the class member variables
        # then determine unique IDs and their records in the extracted data set
        # return map data set that holds full set of records for each unique ID
        # the returned data is clean in terms of not included sets which have zero reference weight
        # though the final data can consist of repeated records and outliers
        self.__get_columns()
        pos_time = positions[0]
        pos_id = positions[1]
        pos_records = positions[2]
        pos_data = positions[3]
        nzlines = self.__count_nzrecords(pos_records)
        self.time = np.empty( shape=(nzlines,1),dtype=np.uint32 )
        #self.time_deb = []
        self.id = np.empty( shape=(nzlines,1),dtype=np.uint32 )
        self.weight = np.empty( shape=(nzlines,1),dtype=np.float32 )
        self.contur_cols = self.nzcolumns-pos_data
        self.mes_data = np.empty( shape=(nzlines,self.nzcolumns-pos_data),dtype=np.float32 )
        i = 0
        with open(self.srcfile) as csv_file:
            csv_reader = csv.reader( csv_file, delimiter = self.dlmt )
            for row in csv_reader:
                if i == 0:
                    i += 1
                else:
                    # do not extract data where reference weight is 0.0
                    if self.__pars_weight( row[pos_records] ) != 0.0:
                        self.time[i-1, 0] = self.__pars_time( row[pos_time] )
                        #self.time_deb.append(row[pos_time])
                        self.id[i-1, 0] = int( row[pos_id] )
                        self.weight[i-1, 0] = self.__pars_weight( row[pos_records] )
                        for j in range( self.nzcolumns-pos_data ):
                            a = row[pos_data+j]
                            a1 = a.strip()
                            a2 = a1.replace(",", ".", 1)
                            self.mes_data[i-1,j] = float(a2)
                        
                        i += 1
        # get list of indexes for each unique ID
        self.__unique()
        # define map 'idata' that should hold extracted data for each ID
        # multiple records are allowed
        idata = {}
        # get the individual data
        for id in self.u_id:
            idata[id] = self.__get_idata( id )        
        # clear memory
        self.__clear()
        return idata

    def __get_idata(self, id):
        # form a data (record) matrix for particular individual
        # return array (for specific ID): ind_data[ num_of_records, (time weight conturs) ]
        ind_data = np.empty( shape=( len( self.unique_id[id] ),self.contur_cols+2 ),dtype=np.float32 )
        i = 0
        #time_data = []
        #print(f'\tNEW id {id}')
        for x in self.unique_id[id]:
            ind_data[i,0] = self.time[x, 0]
            #ind_data[i,0] = float(self.time[x, 0])/86400 # debugging
            #time_data.append(self.time_deb[x])
            ind_data[i,1] = self.weight[x, 0]
            ind_data[i,2:] = self.mes_data[x,:]
            #print(f'\ttime {ind_data[i,0]}, w {ind_data[i,1]}, t(days) {float(ind_data[i,0])/86400}')
            #breakpoint()
            i += 1
        #print(f'\tt(days) {ind_data[:,0]}')
        #print(f'\tt(days) {time_data[:]}')
        #breakpoint()
        return ind_data

    def __clear( self ):
        # Free memory
        del self.time
        del self.id
        del self.weight
        del self.mes_data
        del self.nzcolumns
        del self.unique_id        
        gc.collect()

    def clear( self ):
        # Free memory
        del self.srcfile
        del self.dlmt
        del self.u_id
        gc.collect()

    def rem_outlier( self, data, method, threshold1, threshold2 ):
        
        # entry point for different outlier removal methods
        # data  - array (map) of data sample
        # method - 
        #          zss:    sample based Z-score
        #          zsp:    population based Z-score
        #          zss2:   sample based modified Z-score
        #          cls:    sample based clustering
        #          zs/cls: combination of sample based Z-score
        #                  and sample based clustering
        # threshold1,
        # threshold2     - parameters

        icol1 = 0 # time records
        icol2 = 1 # weight records

        if method == 'zss':
            return self.__zscore_s(data, icol2, threshold1)
        elif method == 'zss2':
            return self.__zscore2_s(data, icol2, threshold1)
        elif method == 'cls':
            return self.__cluster_s(data, icol1, icol2, threshold1, 0)
        elif method == 'zs/cls':
            return self.__zscorecluster_s(data, icol1, icol2, threshold1, threshold2)
        else:
            return data

    def impute( self, data ):
        
        groups = 4

        # loop ove rall IDs
        for i in self.u_id:

            # get size of records for ID = i
            record = len(data[i])-1 # transforming the number of records to array index
            cols = len(data[i][0,:])            
            lContur = math.floor( (cols-2)/groups )

            while record >= 0:
                
                iZer = 0 # number of zero indexes
                fZer = 0 # first zero index
                fFlag = True # flag helped to indicate first zero index

                ig = 0
                i2 = 2
                while i2 < cols:
                    val = data[i][record,i2]
                    if val == 0.0:
                        if fFlag:
                            fZer = i2 # first zero index
                            fFlag = False
                        iZer = iZer + 1 # number of zeros
                        i2 = i2 + 1
                    elif fFlag == False:
                        delta = val/iZer
                        for j in range( fZer+1,i2 ):
                            data[i][record,j] = data[i][record,j-1] + delta
                        fZer = 0
                        fFlag = True
                        iZer = 0
                        ig = ig + lContur
                        i2 = ig + 2                        
                    else:
                        fZer = 0
                        fFlag = True
                        iZer = 0
                        ig = ig + lContur
                        i2 = ig + 2
                
                iZer = 0
                fZer = 0
                fFlag = True

                ig = cols
                i2 = cols - 1
                while i2 > 1:
                    val = data[i][record,i2]
                    if val == 0.0:
                        if fFlag:
                            fZer = i2
                            fFlag = False
                        iZer = iZer + 1
                        i2 = i2 - 1
                    elif fFlag == False:
                        delta = val/iZer
                        for j in range(i2+1,fZer):
                            data[i][record,j] = data[i][record,j-1] - delta
                        fZer = 0
                        fFlag = True
                        iZer = 0
                        ig = ig - lContur
                        i2 = ig - 1
                    else:
                        fZer = 0
                        fFlag = True
                        iZer = 0
                        ig = ig - lContur
                        i2 = ig - 1

                record = record - 1

    def __zscorecluster_s( self, data, t_col, w_col, w_var, w_std ):
        
        # Mixed Clustering / Z-score method:
        #
        # data  - array (map) of data sample
        # t_col - column index with time stamp record
        # w_col - column index with weight record
        # w_var - assumed dayly variability of weight
        # w_std - threshold for outlier,
        #         is the number of STDs above which data is considered as an outlier
        
        newarr = {}
        for i in self.u_id:
            arr = data[i]
            w_ref = arr[0,w_col] # reference weight, (kg)
            t_ref = arr[0,t_col] # reference time, timestemp (seconds)
            tind = 0
            iLow = 0
            iHigh = 0
            low_weight = 0.0
            high_weight = 0.0

            # Clustering:
            # separate data on two distinct sets in relation
            # to dayly weight change - above/below w_var value
            for j in arr[:,w_col]:
                der_w = 0.0
                delta_t = np.abs( float(arr[tind,t_col] - t_ref)/86400 ) # days
                delta_w = np.abs(j - w_ref)
                
                if delta_t != 0.0:
                    der_w = delta_w/delta_t
                
                if der_w < w_var:
                    low_weight = low_weight + j
                    iLow = iLow + 1
                else:
                    high_weight = high_weight + j
                    iHigh = iHigh + 1
                
                tind = tind + 1
            
            # Z-score:
            # here use mean of biggest separated data set, but
            # use std of full (unclustered) data set;
            # we remove all data records above w_std
            if iHigh > iLow:
                m = high_weight/iHigh
            else:
                m = low_weight/iLow
                                
            sig = np.std(arr[:,w_col])
            z0 = []
            
            if sig == 0.0:
                newarr[i] = arr
                continue
            
            for i2 in arr[:,w_col]:
                z0.append(np.abs((i2 - m)/sig))

            z1 = np.empty( shape=(len(z0),1),dtype=np.float64 )
            
            z1[:,0] = z0[:]

            arr = arr[np.all(z1<w_std, axis=1),:]
            
            newarr[i] = arr

        return newarr

    def __cluster_s( self, data, t_col, w_col, w_var, w_mean ):
        
        # Clustering method:
        #
        # data   - array (map) of data sample
        # t_col  - column index with time stamp record
        # w_col  - column index with weight record
        # w_var  - assumed dayly variability of weight
        # w_mean - population mean, is used as criteria
        #          to select a right set among separated (clustered) data sets
        
        newarr = {}
        for i in self.u_id:
            arr_high = []
            arr_low = []
            arr = data[i]
            w_ref = arr[0,w_col] # reference weight, (kg)
            t_ref = arr[0,t_col] # reference time, timestemp (seconds)
            tind = 0
            low_weight = 0.0
            high_weight = 0.0
            
            # Separate data on two distinct sets
            # using a daily weight change as a clustering criteria
            for j in arr[:,w_col]:
                der_w = 0.0
                delta_t = np.abs( float(arr[tind,t_col] - t_ref)/86400 ) # days
                delta_w = np.abs(j - w_ref)
                
                if delta_t != 0.0:
                    der_w = delta_w/delta_t
                
                if der_w < w_var:
                    low_weight = low_weight + j
                    arr_low.append(tind) # we are collecting indexes
                else:
                    high_weight = high_weight + j
                    arr_high.append(tind) # we are collecting indexes
                
                #print(f'\tw_ref {w_ref}, w {j}, der_w {der_w}, delta_w {delta_w}, delta_t {delta_t}, t_ref {float(t_ref)/86400}, t {float(arr[tind,t_col])/86400}')
                #breakpoint()
                tind = tind + 1
            
            # Here we have to choose one right set
            # among the two separated sets;
            # As a criteria we use either:
            # (i) power of clustered sets, so we select the one which has more data
            # or (ii) we use population mean (w_mean parameter) as a criteria, so
            # the data set whose mean is closest to the population mean is selected
            if w_mean == 0:
                # Here we make choise based on size of the separated sets
                if len( arr_high ) > len( arr_low ):
                    z = np.empty( shape=(len(arr_high),arr.shape[1]),dtype=np.float64 )
                    iz = 0
                    for jj in arr_high:
                        z[iz,:] = arr[jj,:]
                        iz = iz + 1                    
                    newarr[i] = z
                    #print(f'\tz {z[:,1]}')
                    #breakpoint()
                else:
                    z = np.empty( shape=(len(arr_low),arr.shape[1]),dtype=np.float64 )
                    iz = 0
                    for jj in arr_low:
                        z[iz,:] = arr[jj,:]
                        iz = iz + 1                    
                    newarr[i] = z
                    #print(f'\tz {z[:,1]}')
                    #breakpoint()
            else:
                # Here we use the population mean
                if len( arr_high ) == 0:
                    h_mean = 0.0
                else:
                    h_mean = high_weight/len( arr_high )
                
                if len( arr_low ) == 0:
                    l_mean = 0.0
                else:
                    l_mean = low_weight/len( arr_low )

                h = np.abs( h_mean - w_mean )
                l = np.abs( l_mean - w_mean )

                if l > h:
                    z = np.empty( shape=(len(arr_high),arr.shape[1]),dtype=np.float64 )
                    iz = 0
                    for jj in arr_high:
                        z[iz,:] = arr[jj,:]
                        iz = iz + 1                    
                    newarr[i] = z
                else:
                    z = np.empty( shape=(len(arr_low),arr.shape[1]),dtype=np.float64 )
                    iz = 0
                    for jj in arr_low:
                        z[iz,:] = arr[jj,:]
                        iz = iz + 1                    
                    newarr[i] = z

        return newarr

    def __zscore_s( self, data, icol, threshold ):
        
        # sample based Z-score method:
        # data      - array (map) of data sample
        # icol      - column index with data record of interest (weight, for example)
        # threshold - threshold for outlier,
        #             is the number of STDs above which data is considered as an outlier

        newarr = {}
        for i in self.u_id:
            arr = data[i]
            m = np.mean(arr[:,icol])
            sig = np.std(arr[:,icol])
            z0 = []
            
            if sig == 0.0:
                newarr[i] = arr
                continue
            
            # calculate deviation from the mean for all records
            for i2 in arr[:,icol]:
                z0.append(np.abs((i2 - m)/sig))

            z1 = np.empty( shape=(len(z0),1),dtype=np.float64 )
            
            z1[:,0] = z0[:]
            
            # remove data above the threshold
            arr = arr[np.all(z1<threshold, axis=1),:]
            
            newarr[i] = arr

        return newarr

    def __zscore2_s( self, data, icol, threshold ):
        
        # sample based modified Z-score method:
        # data      - array (map) of data sample
        # icol      - column index with data record of interest (weight, for example)
        # threshold - threshold for outlier,
        #             is the number of STDs above which data is considered as an outlier

        newarr = {}
        for i in self.u_id:
            arr = data[i]

            md = np.median(arr[:,icol])
            mad = np.median([np.abs(y - md) for y in arr[:,icol]])
            
            if mad == 0.0:
                newarr[i] = arr
                continue
            
            z0 = [ np.abs( 0.6745 * (y - md) / mad ) for y in arr[:,icol]]

            z1 = np.empty( shape=(len(z0),1),dtype=np.float64 )
            
            z1[:,0] = z0[:]
            
            # # remove data above the threshold
            arr = arr[np.all(z1<threshold, axis=1),:]
            
            newarr[i] = arr

        return newarr

    def __zscore_p( self, data ):
        arr = data
        return arr



