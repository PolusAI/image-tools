import pandas, multiprocessing, argparse, logging, matplotlib, copy, imageio
matplotlib.use('agg')
import numpy as np
from multiprocessing import Pool
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from pathlib import Path
import logging                                                                                                                                                            
import sys
import math


global nfeats
global datalen


# Chunk Scale
CHUNK_SIZE = 1024

# Number of Bins for Each Feature
bincount = 100

# DZI file template
DZI = '<?xml version="1.0" encoding="utf-8"?><Image TileSize="' + str(CHUNK_SIZE) + '" Overlap="0" Format="png" xmlns="http://schemas.microsoft.com/deepzoom/2008"><Size Width="{}" Height="{}"/></Image>'

# Initialize the logger    
logging.basicConfig(filename = "logfile", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def pos2pos(data, length, commonratio, min, max):
    replacedata = []
    for val in data:
        log = math.log(val/min, commonratio)
        replacedata.append(float(math.floor(log + 1)))
    return pandas.Series(np.array(replacedata), index = range(0, length))

def neg2neg(data, length, commonratio, min, max):
    replacedata = []
    for val in data:
        log = math.log(val/max, commonratio)
        bin = float(math.floor(log + 1))
        replacedata.append(-1*bin)
    return pandas.Series(np.array(replacedata), index = range(0, length))

def zero2pos(data, length, commonratio, min, max):
    replacedata = [] 
    for val in data:
        if val < min:
            replacedata.append(1.0)
        else:
            log = math.log(val/min, commonratio)
            bin = float(math.floor(log + 2))
            replacedata.append(bin)
    return pandas.Series(np.array(replacedata), index = range(0, length))

def neg2zero(data, length, commonratio, min, max):
    replacedata = []
    for val in data:
        if val > max:
            replacedata.append(-1.0)
            bin = -1.0
        else:
            log = math.log(val/max, commonratio)
            bin = float(math.floor(log + 2))
            replacedata.append(-1*bin)
    return pandas.Series(np.array(replacedata), index = range(0, length))

def neg2pos(data, binsizes, length, commonratio, min, max, alpha, smallbins, largebins):
    replacedata = []
    # print("Number of bins on both sides in function: ", smallbins, largebins, smallbins + largebins)
    # print("Min and Max Data: ", min, max)
    # print("Alpha: ", alpha)
    # print("Commonratio: ", commonratio)
    # print("Binsizes (small, large): ", binsizes)
    binnumber = 0
    for val in data:
        if val > 0:
            if abs(max) > abs(min):
                commonratio = binsizes[1]
                binnumber = largebins
            else:
                commonratio = binsizes[0]
                binnumber = smallbins 
            if val < alpha:
                bin = 1.0
                replacedata.append(1.0)
            elif val == max:
                replacedata.append(binnumber)
            else:
                log = math.log(val/alpha, commonratio)
                bin = float(math.floor(log + 1))
                replacedata.append(abs(bin)) 
                # if abs(max) > abs(min) and replacedata[-1] > largebins:
                #     print("ERROR POS Large: ", val, replacedata[-1])
                # if abs(max) < abs(min) and replacedata[-1] > smallbins:
                #     print("ERROR POS Small: ", val, replacedata[-1])
        else:
            if abs(max) > abs(min):
                commonratio = binsizes[0]
                binnumber = smallbins
            else:
                commonratio = binsizes[1]
                binnumber = largebins
            if val > (-1*alpha):
                replacedata.append(-1.0)
            elif val == min:
                replacedata.append(-1*binnumber)
            else:
                log = math.log(val/(-1*alpha), commonratio)
                bin = -1*float(math.floor(log + 1))
                replacedata.append(bin)
                # if abs(max) > abs(min) and replacedata[-1] > smallbins:
                #     print("ERROR NEG Large: ", val, replacedata[-1])
                # if abs(max) < abs(min) and replacedata[-1] > largebins:
                #     print("ERROR NEG Small: ", val, replacedata[-1])
    return pandas.Series(np.array(replacedata), index = range(0, length))  
                 

def root(root, value):
    return round(value**(1/root), 6)

""" 1. Loading and binning data """
def is_number(value):
    try:
        float(value)
        return True
    except:
        return False

def checkrange(original, transformed, Range, column_bin_sizes, max, min):
    if original >= Range[0] and original <= Range[1]:
        return original, transformed, Range
    else:
        if original > Range[1]:
            transformed = transformed + 1
            diff = original - Range[1]
        if original < Range[0]:
            transformed = transformed - 1
            diff = original - Range[0]


        if transformed > 1:
            Range[0] = column_bin_sizes[-1]**(transformed - 1)
            Range[1] = column_bin_sizes[-1]**(transformed)
        if transformed < -1:
            Range[0] = -1*(column_bin_sizes[0]**(-1*transformed))
            Range[1] = -1*(column_bin_sizes[0]**(-1*transformed - 1))
        if transformed == 1:
            Range[0] = 1
            Range[1] = column_bin_sizes[-1]**(transformed)
        if transformed == -1:
            Range[0] = -1*(column_bin_sizes[0]**(-1*transformed))
            Range[1] = -1
        print(" ")
        print("Difference: ", diff)
        print("Range Edited", Range)
        print("Edited Transformation: ", transformed)
        return checkrange(original, transformed, Range, column_bin_sizes, max, min)

def firstAterm(IQR, datalen):
    return (2*IQR)/(datalen**(1/3))

def load_csv(fpath):
    """ Load a csv and select data
    
    Data is loaded from a csv, and data columns containing numeric values are 
    returned in a pandas Dataframe. The second row of the csv may contain
    column classifiers, so the second row is first loaded and checked to
    determine if the classifiers are present.
    Inputs:
        fpath - Path to csv file
    Outputs:
        data - A pandas Dataframe
        cnames - Names of columns
    """

    # Check if the first row is column coding, and if it is then find valid columns
    data = pandas.read_csv(fpath,nrows=1)
    is_coded = True
    cnames = []
    for ind,fname in zip(range(len(data.columns)),data.columns):
        if data[fname][0] != 'F' and data[fname][0] != 'C':
            is_coded = False
            if is_number(data[fname][0]):
                cnames.append([fname,ind])
            else:
                logging.info('Column {} does not appear to contain numeric values. Not building graphs for this column.'.format(fname))
        elif data[fname][0] == 'F':
            cnames.append([fname,ind])
        else:
            logging.info('Skipping column {} for reason: one hot encodings'.format(fname))
    
    # Load the data
    if is_coded:
        data = pandas.read_csv(fpath,skiprows=[1],usecols=[c[0] for c in cnames])

    else:
        data = pandas.read_csv(fpath,usecols=[c[0] for c in cnames])

    return data, cnames

def initializebins_log(data,column_names):

    linear_index = []
    #bin_sizes = []
    bins = 0
    column_bin_sizes = []
    quartile25to75 = []
    bin_stats = {'size': data.shape,
                 'min': data.min(),
                 'max': data.max(),
                 'twenty5': data.quantile(0.25),
                 'seventy5': data.quantile(0.75)}
    nfeats = bin_stats['size'][1] 
    datalen = bin_stats['size'][0]

    for i in range(0, nfeats):
        quartile25to75.append(bin_stats['seventy5'][i] - bin_stats['twenty5'][i])
        alpha = firstAterm(quartile25to75[-1], datalen)
        if alpha == 0:
            empty = [255] * datalen
            data[column_names[i]] = pandas.Series(np.array(empty), index = range(0, datalen))
            continue
        column_bin_sizes.append(2)
        # if bin_stats['min'][i] > 0 and bin_stats['max'][i] > 0:
        #     column_bin_sizes.append([root(bincount - 1, bin_stats['max'][i]/bin_stats['min'][i])])
        #     data[column_names[i]] = pos2pos(data[column_names[i]], datalen, column_bin_sizes[-1][0], bin_stats['min'][i], bin_stats['max'][i])
        # if bin_stats['min'][i] < 0 and bin_stats['max'][i] < 0:
        #     column_bin_sizes.append([root(bincount - 1, bin_stats['min'][i]/bin_stats['max'][i])])
        #     data[column_names[i]] = neg2neg(data[column_names[i]], datalen, column_bin_sizes[-1][0], bin_stats['min'][i], bin_stats['max'][i])   
        if bin_stats['min'][i] >= 0 and bin_stats['max'][i] > 0:
            column_bin_sizes.append([root(bincount - 2, bin_stats['max'][i]/alpha)])
            data[column_names[i]] = zero2pos(data[column_names[i]], datalen, column_bin_sizes[-1][0], alpha, bin_stats['max'][i])
        if bin_stats['min'][i] < 0 and bin_stats['max'][i] <= 0:
            alpha = alpha * -1
            column_bin_sizes.append([root(bincount - 2, bin_stats['min'][i]/(alpha))])
            data[column_names[i]] = neg2zero(data[column_names[i]], datalen, column_bin_sizes[-1][0], bin_stats['min'][i], alpha)
        if bin_stats['min'][i] < 0 and bin_stats['max'][i] > 0:
            small = 0
            large = 0
            if abs(bin_stats['max'][i]) < abs(bin_stats['min'][i]):
                small = abs(bin_stats['max'][i])
                large = abs(bin_stats['min'][i])
            else:
                small = abs(bin_stats['min'][i])
                large = abs(bin_stats['max'][i])
            ratio = root(bincount + 1, (small*small)/(alpha*large))
            binssmall = int(math.floor(abs(1 + math.log(small/alpha, ratio))))
            binslarge = int(math.floor(abs(1 + math.log(large/alpha, ratio))))
            zfactor = binssmall/binslarge
            binssmall = round(bincount/(2 + 1/zfactor), 0)
            binslarge = binssmall + (bincount - (2*binssmall))
            small_interval = root(binssmall, small/alpha)
            large_interval = root(binslarge, large/alpha)
            column_bin_sizes.append([small_interval, large_interval])        
            data[column_names[i]] = neg2pos(data[column_names[i]], column_bin_sizes[-1], datalen, ratio, bin_stats['min'][i], bin_stats['max'][i], alpha, binssmall, binslarge)
    
    data_ind = pandas.notnull(data)  # Handle NaN values
    data[~data_ind] = bincount + 55          # Handle NaN values
    data = data.astype(np.int16) # cast to save memory
    #data[data==bincount] = bincount - 1         # in case of numerical precision issues
    
    # initialize bins, try to be memory efficient
    nrows = data.shape[0]
    if nrows < 2**8:
        dtype = np.uint8
    elif nrows < 2**16:
        dtype = np.uint16
    elif nrows < 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    bins = np.zeros((nfeats,nfeats,bincount,bincount),dtype=dtype)
 
    return bins, data, data_ind, nfeats, bin_stats
    
    
    #TRANSFORM THE DATA 
    # for i in range(0, nfeats):
    #     if bin_stats['min'][i] > 0:
    #         column_bin_sizes.append([root(bincount - 1, bin_stats['max'][i]/bin_stats['min'][i])])
    #     elif bin_stats['min'][i] == 0:
    #         column_bin_sizes.append([root(bincount - 1, bin_stats['max'][i]/1)])
    #     elif bin_stats['max'][i] < 0:
    #         column_bin_sizes.append([root(bincount - 1, bin_stats['min'[i]]/bin_stats['max'][i])])
    #     elif bin_stats['max'][i] == 0:
    #         column_bin_sizes.append([root(bincount - 1, bin_stats['min'][i]/-1)])
    #     else:
    #         neg_interval = root(bincount/2 - 1, bin_stats['min'][i]/-1)
    #         pos_interval = root(bincount/2 - 1, bin_stats['max'][i]/1)
    #         column_bin_sizes.append([neg_interval, pos_interval])
    #     quartile25to75.append(bin_stats['seventy5'][i] - bin_stats['twenty5'][i])
    #     alpha = firstAterm(quartile25to75[-1], datalen)
    #     print("Data Column: ", column_names[i])
    #     print("A term: ", alpha)
    #     print("Data Range: ", bin_stats['min'][i], bin_stats['max'][i])
    #     for val in data[column_names[i]]:
    #         ogval = val
    #         print("Original Value: ", ogval)
            # print("Interval Size: ", column_bin_sizes[i])
            # #print("Quartile 50 to 75 Range: ", quartile25to75[i])
            # #print("Min and Max Data", bin_stats['min'][i], bin_stats['twenty5'][i], bin_stats['seventy5'][i], bin_stats['max'][i])
            # Range = [0, 0]
            # if val > 1:
            #     if bin_stats['min'][i] < 0:
            #         val = int(math.floor((bincount - math.log(val/bin_stats['max'][i], 1/column_bin_sizes[i][-1])) - bincount/2))
            #         Range[0] = column_bin_sizes[i][-1]**(val - 1)
            #         Range[1] = column_bin_sizes[i][-1]**(val)
            #     elif bin_stats['min'][i] > 0:
            #         #ALL VALUES ARE POSITIVE
            #         val = int(math.floor(math.log(val/bin_stats['min'][i], column_bin_sizes[i][-1]) + 1))
            #         Range[0] = bin_stats['min'][i]*((column_bin_sizes[i][-1])**(val - 1))
            #         Range[1] = bin_stats['min'][i]*((column_bin_sizes[i][-1])**val)
            #     else: 
            #         if val == bin_stats['max'][i]:
            #             val = bincount - 1
            #         val = int(math.floor(math.log(val, column_bin_sizes[i][-1]) + 1))
            #         Range[0] = bin_stats['min'][i]*((column_bin_sizes[i][-1])**(val - 1))
            #         Range[1] = bin_stats['min'][i]*((column_bin_sizes[i][-1])**val) 
            # elif val < -1:
            #     if bin_stats['max'][i] > 0:
            #         val = int(math.floor((bincount - math.log(val/bin_stats['min'][i], 1/column_bin_sizes[i][-1])) - int(bincount/2)))
            #         Range[0] = -1*(column_bin_sizes[i][0]**(val))
            #         Range[1] = -1*(column_bin_sizes[i][0]**(val - 1))
            #         val = -1*val
            #     else:
            #         val = None
            # else:
            #     if bin_stats['min'][i] == 0:
            #         val = 1
            #         Range[1] = 1
            #     elif bin_stats['max'][i] == 0:
            #         val = -1
            #         Range[1] = -1
            #     else:
            #         val = 0
            #         Range[0] = -1
            #         Range[1] = 1

            # print("Transformed Value: ", val)
            # print("Range: ", Range)
            # ogval, val, Range = checkrange(ogval, val, Range, column_bin_sizes[i], bin_stats['max'][i], bin_stats['min'][i])
            # print(" ")
            

    #for i in range(0, nfeats):
    #    print(data[column_names[i]])
    
    #Transform the data
    # copydata = data
    # for i in range(0, nfeats):
    #     for val in data[column_names[i]]:
    #         transformval = val
    #         Binnumber = 0
    #         if val > 0:
    #             print("Column ", column_names[i], " data: ", val)
    #             print("Bin Stats: ", bin_stats['min'][i], bin_stats['max'][i])
    #             print("Bin Sizes: ", column_bin_sizes[i][0], column_bin_sizes[i][-1])
    #             if bin_stats['min'][i] < 0:
    #                 print("Bin", -(bincount - 2)/2, "Range: ", bin_stats['min'][i], bin_stats['min'][i]/column_bin_sizes[i][0])
    #                 if val < bin_stats['min'][i]/column_bin_sizes[i][0]:
    #                     Binnumber = 1
    #                 for space in range(2, int(bincount/2)):
    #                     LowBound = bin_stats['min'][i]/(column_bin_sizes[i][0]**(space - 1))
    #                     UppBound = bin_stats['min'][i]/(column_bin_sizes[i][0]**(space))
    #                     print("Bin", space - bincount/2, "Range: ", bin_stats['min'][i]/(column_bin_sizes[i][0]**(space - 1)), bin_stats['min'][i]/(column_bin_sizes[i][0]**(space)))
    #                     if val > LowBound and val <= UppBound:
    #                         Binnumber = space - bincount/2
    #                 print("Bin 0 Range:  -1.00 1.00")
    #                 for space in range(int(bincount/2) - 1, 0, -1):
    #                     LowBound = bin_stats['max'][i]/(column_bin_sizes[i][-1]**space)
    #                     UppBound = bin_stats['max'][i]/(column_bin_sizes[i][-1]**(space - 1))
    #                     print("Bin", bincount - space - bincount/2, "Range:", bin_stats['max'][i]/(column_bin_sizes[i][-1]**space), bin_stats['max'][i]/(column_bin_sizes[i][-1]**(space - 1)))
    #                     if val > LowBound and val <= UppBound:
    #                         Binnumber = bincount-space - (bincount/2)
    #                 if val > -1 and val <= 1.0:
    #                     Binnumber = 0.0
    #                     transformval = 0
    #                 else:
    #                     transformval = (bincount - math.log(val/bin_stats['max'][i], 1/column_bin_sizes[i][-1])) - bincount/2
    #             elif bin_stats['min'][i] > 0:
    #                 for space in range(bincount -1, 0, -1):
    #                     LowBound = bin_stats['max'][i]/(column_bin_sizes[i][-1]**space)
    #                     UppBound = bin_stats['max'][i]/(column_bin_sizes[i][-1]**(space-1))
    #                     print("All Positive Bin", bincount-space, "Range:", bin_stats['max'][i]/(column_bin_sizes[i][-1]**space), bin_stats['max'][i]/(column_bin_sizes[i][-1]**(space-1)))
    #                     if val > LowBound and val <= UppBound:
    #                         Binnumber = bincount-space
    #                 transformval = math.log(val/bin_stats['min'][i], column_bin_sizes[i][-1]) + 1
    #             else:
    #                 for space in range(bincount -1, 0, -1):
    #                     LowBound = bin_stats['max'][i]/(column_bin_sizes[i][-1]**space)
    #                     UppBound = bin_stats['max'][i]/(column_bin_sizes[i][-1]**(space-1))
    #                     print("All Positive Bin", bincount-space, "Range:", bin_stats['max'][i]/(column_bin_sizes[i][-1]**space), bin_stats['max'][i]/(column_bin_sizes[i][-1]**(space-1)))
    #                     if val > LowBound and val <= UppBound:
    #                         Binnumber = bincount-space
    #                 transformval = math.log(val, column_bin_sizes[i][-1]) + 1
    #             print("Binnumber: ", Binnumber)
    #             print("Tranformed POS data: ", int(math.floor(transformval)))
    #             print(" ")
    #             #val = root(column_bin_sizes[i][0], val/(bin_stats['max'][i]))
    #         elif val < 0:
    #             #val = root(column_bin_sizes[i][-1], val/abs
    #             print("Column ", column_names[i], " data: ", val)
    #             print("Bin Stats: ", bin_stats['min'][i], bin_stats['max'][i])
    #             print("Bin Sizes: ", column_bin_sizes[i][0], column_bin_sizes[i][-1])
    #             if bin_stats['max'][i] > 0:
    #                 print("Bin", -(bincount - 2)/2, "Range: ", bin_stats['min'][i], bin_stats['min'][i]/column_bin_sizes[i][0])
    #                 if val < bin_stats['min'][i]/column_bin_sizes[i][0]:
    #                     Binnumber = int(1 - (bincount/2))
    #                 for space in range(2, int(bincount/2)):
    #                     LowBound = bin_stats['min'][i]/(column_bin_sizes[i][0]**(space - 1))
    #                     UppBound = bin_stats['min'][i]/(column_bin_sizes[i][0]**(space))
    #                     print("Bin", space - bincount/2, "Range: ", bin_stats['min'][i]/(column_bin_sizes[i][0]**(space - 1)), bin_stats['min'][i]/(column_bin_sizes[i][0]**(space)))
    #                     if val > LowBound and val <= UppBound:
    #                         Binnumber = int(space - (bincount/2))
    #                 print("Bin 0 Range:  -1.00 1.00")
    #                 for space in range(int(bincount/2) - 1, 0, -1):
    #                     LowBound = bin_stats['max'][i]/(column_bin_sizes[i][-1]**space)
    #                     UppBound = bin_stats['max'][i]/(column_bin_sizes[i][-1]**(space - 1))
    #                     print("Bin", bincount - space - bincount/2, "Range:", bin_stats['max'][i]/(column_bin_sizes[i][-1]**space), bin_stats['max'][i]/(column_bin_sizes[i][-1]**(space - 1)))
    #                     if val > LowBound and val <= UppBound:
    #                         Binnumber = int(bincount-space - (bincount/2))
    #                 if val > -1 and val <= 1.0:
    #                     Binnumber = 0.0
    #                     transformval = 0
    #                 else:
    #                     transformval = (bincount - math.log(val/bin_stats['min'][i], 1/column_bin_sizes[i][-1])) - int(bincount/2)
    #             else: 
    #                 for space in range(bincount-1, 0, -1):
    #                     print("Bin Range Unknown")
    #                 transformval = math.log(column_bin_sizes[i][-1], abs(val))
    #             print("Binnumber: ", Binnumber)
    #             print("Tranformed NEG data: ", -1*int(math.floor(transformval)))
    #             print(" ")
    #             #val = (root(column_bin_sizes[i][0], val/(bin_stats['min'][i])))*-1
    #         else: 
    #             print("VALUE IS ZERO: ", val)
    #             print(" ")
    #         if Binnumber != math.floor(transformval):
    #             print("ERROR DOES NOT MATCH")
    #             print(" ")




    # for i in range(0, nfeats):
    #     if (bin_stats['min'][i] < 0 and bin_stats['max'][i] > 0):
    #         neg_bins = (np.logspace(np.log(abs(bin_stats['min'][i])), 0, num = bincount/2, base = 10))*-1
    #         pos_bins = np.logspace(0, np.log10(bin_stats['max'][i]), num = bincount/2, base = 10)
    #         bin_sizes.append(np.append(neg_bins, pos_bins))
    #         #bin_sizes.append(np.logspace(np.log10(abs(bin_stats['min'][i])), 0, num = bincount/2, base = 10))
    #         #bin_sizes[-1] = bin_sizes[-1] + (np.logspace(0, np.log10(bin_stats['max'][i]), num = bincount/2, base = 10)) 
    #         #print(bin_sizes[-1])
    #         #print(np.logspace(0, np.log10(bin_stats['max'][i]), num = bincount/2, base = 10))
    #     elif bin_stats['max'][i] < 0:
    #         bin_sizes.append((np.logspace(np.log10(abs(bin_stats['min'][i])), np.log10(abs(bin_stats['max'][i])), num = bincount, base = 10))*-1)
    #     else:
    #         #print("minimum is greater than 0", bin_stats['min'][i])
    #         if bin_stats['min'][i] == 0:
    #             bin_sizes.append(np.logspace(0, np.log10(bin_stats['max'][i]), num = bincount, base = 10))
    #         else:
    #             bin_sizes.append(np.logspace(np.log10(bin_stats['min'][i]), np.log10(bin_stats['max'][i]), num = bincount, base = 10))


    # for i in range(0, len(bin_sizes)):
    #     print(column_names[i])
    #     print(bin_sizes[i], bin_stats['min'][i], bin_stats['max'][i])
    #     print(bin_sizes[i][1]/bin_sizes[i][0], bin_sizes[i][2]/bin_sizes[i][1])
    #     print(bin_sizes[i][-1]/bin_sizes[i][-2], bin_sizes[i][-2]/bin_sizes[i][-3])
    #     print('\n')
    

def initializebins(data, column_names):
   
    # Get basic column statistics and bin sizes
    nfeats = len(column_names)
    bin_stats = {'min': data.min(),
                 'max': data.max()}
    #print(bin_stats)
    column_bin_size = (bin_stats['max'] * (1 + 10**-6) - bin_stats['min'])/bincount #might be different for log scale??

    # Transform data into bin positions for fast binning
    data = ((data - bin_stats['min'])/column_bin_size).apply(np.floor)
    data_ind = pandas.notnull(data)  # Handle NaN values
    data[~data_ind] = bincount + 55          # Handle NaN values
    data = data.astype(np.uint16) # cast to save memory
    data[data==bincount] = bincount - 1         # in case of numerical precision issues

    # initialize bins, try to be memory efficient
    nrows = data.shape[0]
    if nrows < 2**8:
        dtype = np.uint8
    elif nrows < 2**16:
        dtype = np.uint16
    elif nrows < 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    bins = np.zeros((nfeats,nfeats,bincount,bincount),dtype=dtype)
 
    return bins, data, data_ind, nfeats, bin_stats

def bin_data_log(data,column_names):
    """ Bin the data
    
    Data from a pandas Dataframe is binned in two dimensions. Binning is performed by
    binning data in one column along one axis and another column is binned along the
    other axis. All combinations of columns are binned without repeats or transposition.
    There are only 20 bins in each dimension, and each bin is 1/20th the size of the
    difference between the maximum and minimum of each column.
    Inputs:
        data - A pandas Dataframe, with nfeats number of columns
        column_names - Names of Dataframe columns
    Outputs:
        bins - A numpy matrix that has shape (nfeats,nfeats,bincount,bincount)
        bin_feats - A list containing the minimum and maximum values of each column
        linear_index - Numeric value of column index from original csv
    """


    bins, data, data_ind, nfeats, bin_stats = initializebins_log(data, column_names)
    #print(data)
    # Create a linear index for feature bins
    linear_index = []
    Datapoints = []
    Ind2 = []
    Ind2zero = []
    Position = []
    datalen = bin_stats['size'][0]
    
    
    
    for feat1 in range(nfeats):
        name1 = column_names[feat1]
        feat1_tf = data[name1]
        for feat2 in range(feat1 + 1, nfeats):
            Datapoints.append([])
            Ind2.append([])
            Ind2zero.append([])
            Position.append([])
            name2 = column_names[feat2]
            feat2_tf = data[name2]
            
            feat1_tf = feat1_tf[data_ind[name1] & data_ind[name2]]
            feat2_tf = feat2_tf[data_ind[name1] & data_ind[name2]]
            
            # print(feat1_tf, feat2_tf)
            # print(" \n")
            for val in range(0, datalen):
                #print((feat1_tf[val], feat2_tf[val]))
                Datapoints[-1].append((feat1_tf[val], feat2_tf[val]))
            Datapoints[-1] = sorted(Datapoints[-1], key=lambda element: (element[0], element[1]))
            for item in range(0, datalen - 1):
                #print(Datapoints[-1][item], Datapoints[-1][item+1], Datapoints[-1][item][0] - Datapoints[-1][item+1][0], Datapoints[-1][item][1] - Datapoints[-1][item+1][1])
                Ind2[-1].append([Datapoints[-1][item][0] - Datapoints[-1][item+1][0], Datapoints[-1][item][1] - Datapoints[-1][item+1][1]])
                if Ind2[-1][-1] != [0, 0]:
                    Ind2zero[-1].append(Ind2[-1][-1])
                    Position[-1].append(item)
            UniqueData = list(set(Datapoints[-1]))
            UniqueData = sorted(UniqueData, key = lambda element: (element[0], element[1]))

            # print("Length of No Zeros List: ", len(Ind2zero[-1]), Ind2zero[-1])
            # print("Length of Position List: ", len(Position[-1]), Position[-1])
            # print("Length of Unique Items: ", len(UniqueData), UniqueData)            
            # print("Length of Indexes: ", len(Ind2[-1]), Ind2[-1])
            # print("Length of Data: ", len(Datapoints[-1]), Datapoints[-1])
            
            for i in range(0, len(UniqueData)):
                if i == 0:
                    print("There are", str(Position[-1][i] + 1), "observations of", str(UniqueData[i]))
                elif (i == (len(UniqueData)-1)):
                    print("There are", str(datalen - Position[-1][i - 1]), "observations of", str(UniqueData[i]))
                else:
                    print("There are", str(Position[-1][i] - Position[-1][i - 1]), "observations of", str(UniqueData[i]))    
            print(name1, name2, "NEXT")
            print(" ")
            
    # Bin the data
    # for feat1 in range(nfeats):
    #     name1 = column_names[feat1]
    #     feat1_tf = data[name1] * bincount   # Convert to linear matrix index
    #     #print(feat1_tf)
    #     for feat2 in range(feat1+1,nfeats):
    #         name2 = column_names[feat2]
            
    #         # Remove all NaN values
    #         feat2_tf = data[name2]
    #         feat2_tf = feat2_tf[data_ind[name1] & data_ind[name2]]
            
    #         if feat2_tf.size<=1:
    #             continue
            
    #         # sort linear matrix indices
    #         feat2_sort = np.sort(feat1_tf[data_ind[name1] & data_ind[name2]] + feat2_tf)
    #         #print(feat2_sort)
    #         # Do math to get the indices
    #         ind2 = np.diff(feat2_sort)                       
    #         ind2 = np.nonzero(ind2)[0]                       # nonzeros are cumulative sum of all bin values
    #         ind2 = np.append(ind2,feat2_sort.size-1)
    #         # print(feat2_sort.shape)
    #         rows = (feat2_sort[ind2]/bincount).astype(np.int8)   # calculate row from linear index
    #         cols = np.mod(feat2_sort[ind2],bincount)              # calculate column from linear index
    #         # for item in rows:
    #         #     print(name1, name2, rows[item], cols[item])
    #         counts = np.diff(ind2)                           # calculate the number of values in each bin
    #         bins[feat1,feat2,rows[0],cols[0]] = ind2[0] + 1
    #         bins[feat1,feat2,rows[1:],cols[1:]] = counts
    #         linear_index.append([feat1,feat2])
    return bins, bin_stats, linear_index

def bin_data(data,column_names):
    """ Bin the data
    
    Data from a pandas Dataframe is binned in two dimensions. Binning is performed by
    binning data in one column along one axis and another column is binned along the
    other axis. All combinations of columns are binned without repeats or transposition.
    There are only 20 bins in each dimension, and each bin is 1/20th the size of the
    difference between the maximum and minimum of each column.
    Inputs:
        data - A pandas Dataframe, with nfeats number of columns
        column_names - Names of Dataframe columns
    Outputs:
        bins - A numpy matrix that has shape (nfeats,nfeats,bincount,bincount)
        bin_feats - A list containing the minimum and maximum values of each column
        linear_index - Numeric value of column index from original csv
    """


    bins, data, data_ind, nfeats, bin_stats = initializebins(data, column_names)

    # Create a linear index for feature bins
    linear_index = []

    # Bin the data
    for feat1 in range(nfeats):
        name1 = column_names[feat1]
        feat1_tf = data[name1] * bincount   # Convert to linear matrix index

        for feat2 in range(feat1+1,nfeats):
            name2 = column_names[feat2]
            
            # Remove all NaN values
            feat2_tf = data[name2]
            feat2_tf = feat2_tf[data_ind[name1] & data_ind[name2]]
            
            if feat2_tf.size<=1:
                continue
            
            # sort linear matrix indices
            feat2_sort = np.sort(feat1_tf[data_ind[name1] & data_ind[name2]] + feat2_tf)
            
            # Do math to get the indices
            ind2 = np.diff(feat2_sort)                       
            ind2 = np.nonzero(ind2)[0]                       # nonzeros are cumulative sum of all bin values
            ind2 = np.append(ind2,feat2_sort.size-1)
            # print(feat2_sort.shape)
            rows = (feat2_sort[ind2]/bincount).astype(np.uint8)   # calculate row from linear index
            cols = np.mod(feat2_sort[ind2],bincount)              # calculate column from linear index
            counts = np.diff(ind2)                           # calculate the number of values in each bin
            bins[feat1,feat2,rows[0],cols[0]] = ind2[0] + 1
            bins[feat1,feat2,rows[1:],cols[1:]] = counts
            linear_index.append([feat1,feat2])
    return bins, bin_stats, linear_index

""" 2. Plot Generation """

# Tick formatting to mimick D3
def format_ticks(fmin,fmax,nticks):
    """ Generate tick labels
    Polus Plots uses D3 to generate the plots. This function tries to mimic
    the formatting of tick labels. Tick labels have a fixed width, and in
    place of using scientific notation a scale prefix is appended to the end
    of the number. See _prefix comments to see the suffixes that are used.
    Numbers that are larger or smaller than 10**24 or 10**-24 respectively
    are not handled and may throw an error. Values outside of this range
    do not currently have an agreed upon prefix in the measurement science
    community.
    Inputs:
        fmin - the minimum tick value
        fmax - the maximum tick value
        nticks - the number of ticks
    Outputs:
        fticks - a list of strings containing formatted tick labels
    """
    _prefix = {-24: 'y',  # yocto
               -21: 'z',  # zepto
               -18: 'a',  # atto
               -15: 'f',  # femto
               -12: 'p',  # pico
                -9: 'n',  # nano
                -6: 'u',  # micro
                -3: 'm',  # mili
                 0: ' ',
                 3: 'k',  # kilo
                 6: 'M',  # mega
                 9: 'G',  # giga
                12: 'T',  # tera
                15: 'P',  # peta
                18: 'E',  # exa
                21: 'Z',  # zetta
                24: 'Y',  # yotta
                }
    out = [t for t in np.arange(fmin,fmax,(fmax-fmin)/(nticks-1))]
    out.append(fmax)
    scale = np.log10(np.abs(out))   #Ignore Warning.  Run python -W ignore main.py on command prompt
    scale[np.isinf(scale)] = 0
    #logger.info("SCALE(INF): " + str(scale))
    scale_order = np.int8(3*np.sign(scale)*np.int8(scale/3))
    fticks = []
    for i in range(nticks):
        fticks.append('{:{width}.{prec}f}'.format(out[i]/10**scale_order[i],
                                                  width=3,
                                                  prec=2-np.mod(np.int8(scale[i]),3)) + _prefix[scale_order[i]])
    return fticks

def get_cmap():
    # Create a custom colormap to mimick Polus Plots

    cmap_values = [[1.0,1.0,1.0,1.0]]
    cmap_values.extend([[r/255,g/255,b/255,1] for r,g,b in zip(np.arange(0,255,2),
                                                            np.arange(153,255+1/128,102/126),
                                                            np.arange(34+1/128,0,-34/126))])
    cmap_values.extend([[r/255,g/255,b/255,1] for r,g,b in zip(np.arange(255,136-1/128,-119/127),
                                                            np.arange(255,0,-2),
                                                            np.arange(0,68+1/128,68/127))])
    cmap = ListedColormap(cmap_values)

    return cmap

def gen_plot(col1,
             col2,
             bins,
             column_names,
             bin_stats,
             fig,
             ax,
             data):
    """ Generate a heatmap
    Generate a heatmap of data for column 1 against column 2.
    Inputs:
        col1 - the column plotted on the y-axis
        col2 - column plotted on the x-axis
        bins - bin data generated by bin_data()
        column_names - list of column names
        bin_stats - a list containing the min,max values of each column
        fig - pregenerated figure
        ax - pregenerated axis
        data - pregenerated heatmap bbox artist
    Outputs:
        hmap - A numpy array containing pixels of the heatmap
    """
    if col2>col1:
        d = np.squeeze(bins[col1,col2,:,:])
        r = col1
        c = col2
    elif col2<col1:
        d = np.transpose(np.squeeze(bins[col2,col1,:,:]))
        r = col2
        c = col1
    else:
        d = np.zeros((bincount,bincount))
        r = col1
        c = col2
        
    data.set_data(d)
    data.set_clim(np.min(d),np.max(d))

    ax.set_xlabel(column_names[c])
    ax.set_xticklabels(format_ticks(bin_stats['min'][column_names[c]],bin_stats['max'][column_names[c]],11),
                       rotation=45)
    ax.set_ylabel(column_names[r])
    ax.set_yticklabels(format_ticks(bin_stats['min'][column_names[r]],bin_stats['max'][column_names[r]],11))
    
    fig.canvas.draw()

    hmap = np.array(fig.canvas.renderer.buffer_rgba())
    
    return hmap

def get_default_fig(cmap):
    """ Generate a default figure, axis, and heatmap artist
    Generate a figure and draw an empty graph with useful settings for repeated
    drawing of new figures. By passing the existing figure, axis, and heatmap
    artist to the plot generator, many things do not need to be drawn from
    scratch. This decreases the plot drawing time by a factor of 2-3 times.
    Inputs:
        cmap - the heatmap colormap
    Outputs:
        fig - A reference to the figure object
        ax - A reference to the axis object
        data - A reference to the heatmap artist
    """
    fig, ax = plt.subplots(dpi=int(CHUNK_SIZE/4),figsize=(4,4),tight_layout={'h_pad':1,'w_pad':1})
    data = ax.pcolorfast(np.zeros((CHUNK_SIZE,CHUNK_SIZE),np.uint64),cmap=cmap)
    ticks = [t for t in range(0,199,20)]
    ticks.append(199)
    ax.set_xlim(0,199)
    ax.set_ylim(0,199)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(" ")
    ax.set_xticklabels(["     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     "],
                    rotation=45)
    ax.set_ylabel(" ")
    ax.set_yticklabels(["     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     ",
                        "     "])
    fig.canvas.draw()
    return fig, ax, data

""" 3. Pyramid generation functions """

def _avg2(image):
    """ Average pixels with optical field of 2x2 and stride 2 """
    
    # Convert 32-bit pixels to prevent overflow during averaging
    image = image.astype(np.uint32)
    
    # Get the height and width of each image to the nearest even number
    y_max = image.shape[0] - image.shape[0] % 2
    x_max = image.shape[1] - image.shape[1] % 2
    
    # Perform averaging
    avg_img = np.zeros(np.ceil([image.shape[0]/2,image.shape[1]/2,image.shape[2]]).astype(np.uint32))
    for z in range(4):
        avg_img[0:int(y_max/2),0:int(x_max/2),z]= (image[0:y_max-1:2,0:x_max-1:2,z] + \
                                                   image[1:y_max:2,0:x_max-1:2,z] + \
                                                   image[0:y_max-1:2,1:x_max:2,z] + \
                                                   image[1:y_max:2,1:x_max:2,z]) / 4
        
    # The next if statements handle edge cases if the height or width of the image has an
    # odd number of pixels
    if y_max != image.shape[0]:
        for z in range(3):
            avg_img[-1,:int(x_max/2),z] = (image[-1,0:x_max-1:2,z] + \
                                           image[-1,1:x_max:2,z]) / 2
    if x_max != image.shape[1]:
        for z in range(4):
            avg_img[:int(y_max/2),-1,z] = (image[0:y_max-1:2,-1,z] + \
                                           image[1:y_max:2,-1,z]) / 2
    if y_max != image.shape[0] and x_max != image.shape[1]:
        for z in range(4):
            avg_img[-1,-1,z] = image[-1,-1,z]
    return avg_img

def metadata_to_graph_info(bins,outPath,outFile):
    
    # Create an output path object for the info file
    op = Path(outPath).joinpath("{}.dzi".format(outFile))
    
    # create an output path for the images
    of = Path(outPath).joinpath('{}_files'.format(outFile))
    of.mkdir(exist_ok=True)
    
    # Get metadata info from the bfio reader
    ngraphs = len(linear_index)
    rows = np.ceil(np.sqrt(ngraphs))
    cols = np.round(np.sqrt(ngraphs))
    sizes = [cols*CHUNK_SIZE,rows*CHUNK_SIZE]
    
    # Calculate the number of pyramid levels
    num_scales = np.ceil(np.log2(rows*CHUNK_SIZE)).astype(np.uint8)
    
    # create a scales template, use the full resolution
    scales = {
        "size":sizes,
        "key": num_scales
    }
    
    # initialize the json dictionary
    info = {
        "scales": [scales],       # Will build scales belows
        "rows": rows,
        "cols": cols
    }
    
    # create the information for each scale
    for i in range(1,num_scales+1):
        previous_scale = info['scales'][-1]
        current_scale = copy.deepcopy(previous_scale)
        current_scale['key'] = str(num_scales - i)
        current_scale['size'] = [int(np.ceil(previous_scale['size'][0]/2)),int(np.ceil(previous_scale['size'][1]/2))]
        info['scales'].append(current_scale)
    
    # write the dzi file
    with open(op,'w') as writer:
        writer.write(DZI.format(int(info['cols']*CHUNK_SIZE),int(info['rows']*CHUNK_SIZE)))
    
    return info

# The following function builds the image pyramid at scale S by building up only the necessary information
# at high resolution layers of the pyramid. So, if 0 is the original resolution of the image, getting a tile
# at scale 2 will generate only the necessary information at layers 0 and 1 to create the desired tile at
# layer 2. This function is recursive and can be parallelized.
def _get_higher_res(S,info,outpath,out_file,X=None,Y=None):

    # Get the scale info
    scale_info = None
    logger.info("S: " + str(S))
    logger.info(info['scales'])
    for res in info['scales']:
        if int(res['key'])==S:
            scale_info = res
            break
    if scale_info==None:
        ValueError("No scale information for resolution {}.".format(S))
        
    if X == None:
        X = [0,scale_info['size'][0]]
    if Y == None:
        Y = [0,scale_info['size'][1]]
    logger.info(str(X) + str(Y))
    # Modify upper bound to stay within resolution dimensions
    if X[1] > scale_info['size'][0]:
        X[1] = scale_info['size'][0]
    if Y[1] > scale_info['size'][1]:
        Y[1] = scale_info['size'][1]
    logger.info(str(X) + str(Y))
    
    # Initialize the output
    image = np.zeros((int(Y[1]-Y[0]),int(X[1]-X[0]),4),dtype=np.uint8)
    logger.info(image.shape)
    
    
    # If requesting from the lowest scale, then just generate the graph
    if S==int(info['scales'][0]['key']):
        index = int((int(Y[0]/CHUNK_SIZE) + int(X[0]/CHUNK_SIZE) * info['rows']))
        if index>=len(linear_index):
            image = np.ones((CHUNK_SIZE,CHUNK_SIZE,4),dtype=np.uint8) * (bincount + 55)
        else:
            image = gen_plot(linear_index[index][0],
                             linear_index[index][1],
                             bins,
                             column_names,
                             bin_stats,
                             fig,
                             ax,
                             data)
    else:
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]]]
        logger.info("SUBGRID DIMENSIONS: " + str(subgrid_dims))
        
        for dim in subgrid_dims:
            while dim[1]-dim[0] > CHUNK_SIZE:
                dim.insert(1,dim[0] + ((dim[1] - dim[0]-1)//CHUNK_SIZE) * CHUNK_SIZE)
        logger.info("SUBGRID DIMENSIONS ADDED: " + str(subgrid_dims))
        

        for y in range(0,len(subgrid_dims[1])-1):
            y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
            logger.info("Y index: " + str(y_ind))
            y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
            logger.info("Y index: " + str(y_ind))
            for x in range(0,len(subgrid_dims[0])-1):
                x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                logger.info("X index: " + str(x_ind))
                x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                logger.info("X index: " + str(x_ind))
                logger.info("What X would be: " + str(subgrid_dims[0][x:x+2]))
                logger.info("What Y would be: " + str(subgrid_dims[0][y:y+2]))
                if S==(info['scales'][0]['key'] - 5): #to use multiple processors to compute faster.
                    sub_image = _get_higher_res_par(S+1,
                                                   info,
                                                   outpath,
                                                   out_file,
                                                   X=subgrid_dims[0][x:x+2],
                                                   Y=subgrid_dims[1][y:y+2])
                else:
                    sub_image = _get_higher_res(S+1,
                                               info,
                                               outpath,
                                               out_file,
                                               X=subgrid_dims[0][x:x+2],
                                               Y=subgrid_dims[1][y:y+2])
                # sub_image = _get_higher_res(S+1,
                #                                 info,
                #                                 outpath,
                #                                 X=subgrid_dims[0][x:x+2],
                #                                 Y=subgrid_dims[1][y:y+2])
                
                image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],:] = _avg2(sub_image)
                del sub_image

    # Write the chunk
    outpath = Path(outpath).joinpath('{}_files'.format(out_file),str(S))
    outpath.mkdir(exist_ok=True)
    imageio.imwrite(outpath.joinpath('{}_{}.png'.format(int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE))),image,format='PNG-FI',compression=1)
    logger.info('Finished building tile (scale,X,Y): ({},{},{})'.format(S,int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE)))
    return image

# This function performs the same operation as _get_highe_res, except it uses multiprocessing to grab higher
# resolution layers at a specific layer.
def _get_higher_res_par(S,info,outpath,out_file,X=None,Y=None):
    # Get the scale info
    scale_info = None
    for res in info['scales']:
        if int(res['key'])==S:
            scale_info = res
            break
    if scale_info==None:
        ValueError("No scale information for resolution {}.".format(S))
        
    if X == None:
        X = [0,scale_info['size'][0]]
    if Y == None:
        Y = [0,scale_info['size'][1]]
    
    # Modify upper bound to stay within resolution dimensions
    if X[1] > scale_info['size'][0]:
        X[1] = scale_info['size'][0]
    if Y[1] > scale_info['size'][1]:
        Y[1] = scale_info['size'][1]
    
    # Initialize the output
    image = np.zeros((Y[1]-Y[0],X[1]-X[0],4),dtype=np.uint8)
    
    # If requesting from the lowest scale, then just generate the graph
    if S==int(info['scales'][0]['key']):
        index = (int(Y[0]/CHUNK_SIZE) + int(X[0]/CHUNK_SIZE) * info['rows'])
        if index>=len(linear_index):
            image = np.ones((CHUNK_SIZE,CHUNK_SIZE,4),dtype=np.uint8) * (bincount + 55)
        else:
            image = gen_plot(linear_index[index][0],
                             linear_index[index][1],
                             bins,
                             column_names,
                             bin_stats,
                             fig,
                             ax,
                             data)
    else:
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]]]
        for dim in subgrid_dims:
            while dim[1]-dim[0] > CHUNK_SIZE:
                dim.insert(1,dim[0] + ((dim[1] - dim[0]-1)//CHUNK_SIZE) * CHUNK_SIZE)
        
        subgrid_images = []
        
        with Pool(processes=np.min(4,initial=multiprocessing.cpu_count())) as pool:
            for y in range(0,len(subgrid_dims[1])-1):
                y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
                y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
                for x in range(0,len(subgrid_dims[0])-1):
                    x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                    x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                    subgrid_images.append(pool.apply_async(_get_higher_res,(S+1,
                                                                           info,
                                                                           outpath,
                                                                           out_file,
                                                                           subgrid_dims[0][x:x+2],
                                                                           subgrid_dims[1][y:y+2])))
                    
            for y in range(0,len(subgrid_dims[1])-1):
                y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
                y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
                for x in range(0,len(subgrid_dims[0])-1):
                    x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                    x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                    image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],:] = _avg2(subgrid_images[y*(len(subgrid_dims[0])-1) + x].get())
        del subgrid_images

    # Write the chunk
    outpath = Path(outpath).joinpath('{}_files'.format(out_file),str(S))
    outpath.mkdir(exist_ok=True)
    imageio.imwrite(outpath.joinpath('{}_{}.png'.format(int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE))),image,format='PNG-FI',compression=1)
    logger.info('Finished building tile (scale,X,Y): ({},{},{})'.format(S,int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE)))
    return image

def write_csv(cnames,linear_index,f_info,out_path,out_file):
    header = 'dataset_id, x_axis_id, y_axis_id, x_axis_name, y_axis_name, title, length, width, global_row, global_col\n'
    line = '{:d}, {:d}, {:d}, {:s}, {:s}, default title, {:d}, {:d}, {:d}, {:d}\n'
    l_ind = 0
    with open(str(Path(out_path).joinpath(out_file+'.csv').absolute()),'w') as writer:
        writer.write(header)
        for ind in linear_index:
            writer.write(line.format(1,
                                     cnames[ind[1]][1],
                                     cnames[ind[0]][1],
                                     cnames[ind[1]][0],
                                     cnames[ind[0]][0],
                                     CHUNK_SIZE,
                                     CHUNK_SIZE,
                                     int(np.mod(l_ind,f_info['rows'])),
                                     int(l_ind/f_info['rows'])))
            l_ind += 1
        
if __name__=="__main__":
    
    
    """ Initialize argument parser """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Build an image pyramid from data in a csv file.')

    """ Define the arguments """
    parser.add_argument('--inpDir',               # input image collection directory
                        dest='inpDir',
                        type=str,
                        help='Path to input images.',
                        required=True)
    parser.add_argument('--outDir',               # Pyramid directory
                        dest='outDir',
                        type=str,
                        help='The output directory for the flatfield images.',
                        required=True)
    
    """ Get the input arguments """
    args = parser.parse_args()

    input_path = args.inpDir
    output_path = Path(args.outDir)

    logger.info('inpDir = {}'.format(input_path))
    logger.info('outDir = {}'.format(output_path))

    # Get the path to each csv file in the collection
    input_files = [str(f.absolute()) for f in Path(input_path).iterdir() if ''.join(f.suffixes)=='.csv']

    for f in input_files:
        # Set the file path folder
        folder = Path(f)
        folder = folder.name.replace('.csv','')
        logger.info('Processing: {}'.format(folder))
        
        # Load the data
        logger.info('Loading csv: {}'.format(f))
        data, cnames = load_csv(f)
        column_names = data.columns
        logger.info('Done loading csv!')

        # Bin the data
        logger.info('Binning data for {} features...'.format(column_names.size))
        bins_log, bin_stats_log, linear_index_log = bin_data_log(data, column_names)
        bins, bin_stats, linear_index = bin_data(data,column_names)
        
        logger.info('Done!')
        del data    # get rid of the original data to save memory

        # Generate the default figure components
        logger.info('Generating colormap and default figure...')
        cmap = get_cmap()
        fig, ax, data = get_default_fig(cmap)
        logger.info('Done!')

        # Generate the dzi file
        logger.info('Generating pyramid metadata...')
        info = metadata_to_graph_info(bins,output_path,folder)
        logger.info('Done!')
        
        logger.info('Writing layout file...!')
        write_csv(cnames,linear_index,info,output_path,folder)
        logger.info('Done!')

        # Create the pyramid
        logger.info('Building pyramid...')
        image = _get_higher_res(0,info,output_path,folder)
