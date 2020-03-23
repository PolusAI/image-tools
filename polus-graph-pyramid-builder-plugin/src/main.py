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
import decimal
from decimal import Decimal
from textwrap import wrap
import time
import pickle
import os 


# Chunk Scale
CHUNK_SIZE = 512

# Number of Bins for Each Feature
bincount = 20 #MUST BE EVEN NUMBER

# DZI file template
DZI = '<?xml version="1.0" encoding="utf-8"?><Image TileSize="' + str(CHUNK_SIZE) + '" Overlap="0" Format="png" xmlns="http://schemas.microsoft.com/deepzoom/2008"><Size Width="{}" Height="{}"/></Image>'

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

#Transforms the DataFrame that Range from a Negative Number to a Positive Number
def dfneg2pos(data, alpha, datmin, datmax):

    yaxis = []
    commonratios = []
    alphas = []
    for col in data.columns:  
        small = 0 
        large = 0
        posbigger = True

        # Determining which side requires more bins. 
        if abs(datmax[col]) > abs(datmin[col]):
            small = abs(datmin[col])
            large = abs(datmax[col])
        else:
            small = abs(datmax[col])
            large = abs(datmin[col])
            posbigger = False

        ratio = ((small*small)/(alpha[col]*large))**(1/(bincount + 1))
        binssmall = np.floor(abs(1 + np.log(small/alpha[col])/np.log(ratio)))
        binslarge = np.floor(abs(1 + np.log(large/alpha[col])/np.log(ratio)))
        zfactor = binssmall/binslarge
        binssmall = round(bincount/(2 + 1/zfactor), 0)
        binslarge = binssmall + (bincount - (2*binssmall))
        small_interval = (small/alpha[col])**(1/binssmall)
        large_interval = (large/alpha[col])**(1/binslarge)
            
        commonratios.append([small_interval, large_interval])
        alphas.append(alpha[col])

        datacol = data[col].to_numpy()
        
        # Each value in the Range falls under one of the following conditions
        condition1 = np.asarray((datacol > 0) & (datacol < alpha[col])).nonzero()
        condition2 = np.asarray((datacol > 0) & (datacol == datmax[col])).nonzero()
        condition3 = np.asarray((datacol > 0) & (datacol >= alpha[col])).nonzero()
        condition4 = np.asarray((datacol < 0) & (datacol > -1*alpha[col])).nonzero()
        condition5 = np.asarray((datacol < 0) & (datacol == datmin[col])).nonzero()
        condition6 = np.asarray((datacol < 0) & (datacol <= -1*alpha[col])).nonzero()
        condition7 = np.asarray(datacol == 0).nonzero()

        if posbigger == True:
            yaxis.append(binssmall)
            logged3 = np.log(datacol[condition3]/alpha[col])/np.log(large_interval) + 1
            floored3 = np.float64(np.floor(logged3))
            absolute3 = abs(floored3) + binssmall
            datacol[condition3] = absolute3

            logged6 = np.log(datacol[condition6]/(-1*alpha[col]))/np.log(small_interval) + 1
            floored6 = np.float64(np.floor(logged6))
            absolute6 = -1*abs(floored6) + binssmall
            datacol[condition6] = absolute6

            datacol[condition1] = 1 + binssmall

            datacol[condition2] = bincount

            datacol[condition4] = -1 + binssmall

            datacol[condition5] = 0

            datacol[condition7] = -1 + binssmall

        else:
            yaxis.append(binslarge)
            logged3 = np.log(datacol[condition3]/alpha[col])/np.log(small_interval) + 1
            floored3 = np.float64(np.floor(logged3))
            absolute3 = abs(floored3) + binslarge
            datacol[condition3] = absolute3

            logged6 = np.log(datacol[condition6]/(-1*alpha[col]))/np.log(large_interval) + 1
            floored6 = np.float64(np.floor(logged6))
            absolute6 = -1*abs(floored6) + binslarge
            datacol[condition6] = absolute6

            datacol[condition1] = 1 + binslarge

            datacol[condition2] = bincount

            datacol[condition4] = -1 + binslarge

            datacol[condition5] = 0

            datacol[condition7] = -1 + binslarge


    return yaxis, alphas, commonratios, data

def neg2pos(val, binsizes, length, commonratio, datmin, datmax, alpha, smallbins, largebins):
    tranval = val.to_numpy()
    if abs(datmax) > abs(datmin):

        condition1 = np.asarray((tranval > 0) & (tranval < alpha)).nonzero()
        condition2 = np.asarray((tranval > 0) & (tranval == datmax)).nonzero()
        condition3 = np.asarray((tranval > 0) & (tranval >= alpha)).nonzero()
        condition4 = np.asarray((tranval < 0) & (tranval > -1*alpha)).nonzero()
        condition5 = np.asarray((tranval < 0) & (tranval == datmin)).nonzero()
        condition6 = np.asarray((tranval < 0) & (tranval <= -1*alpha)).nonzero()
        condition7 = np.asarray(tranval == 0).nonzero()

        logged3 = np.log(tranval[condition3]/alpha)/np.log(binsizes[1]) + 1
        floored3 = np.float64(np.floor(logged3))
        absolute3 = abs(floored3) + smallbins
        tranval[condition3] = absolute3

        logged6 = np.log(tranval[condition6]/(-1*alpha))/np.log(binsizes[0]) + 1
        floored6 = np.float64(np.floor(logged6))
        absolute6 = -1*abs(floored6) + smallbins
        tranval[condition6] = absolute6

        tranval[condition1] = 1 + smallbins

        tranval[condition2] = bincount

        tranval[condition4] = -1 + smallbins

        tranval[condition5] = 0

        tranval[condition7] = -1 + smallbins

    else:

        condition1 = np.asarray((tranval > 0) & (tranval < alpha)).nonzero()
        condition2 = np.asarray((tranval > 0) & (tranval == datmax)).nonzero()
        condition3 = np.asarray((tranval > 0) & (tranval >= alpha)).nonzero()
        condition4 = np.asarray((tranval < 0) & (tranval > -1*alpha)).nonzero()
        condition5 = np.asarray((tranval < 0) & (tranval == datmin)).nonzero()
        condition6 = np.asarray((tranval < 0) & (tranval <= -1*alpha)).nonzero()
        condition7 = np.asarray(tranval == 0).nonzero()

        logged3 = np.log(tranval[condition3]/alpha)/np.log(binsizes[0]) + 1
        floored3 = np.float64(np.floor(logged3))
        absolute3 = abs(floored3) + largebins
        tranval[condition3] = absolute3

        logged6 = np.log(tranval[condition6]/(-1*alpha))/np.log(binsizes[0]) + 1
        floored6 = np.float64(np.floor(logged6))
        absolute6 = -1*abs(floored6) + largebins
        tranval[condition6] = absolute6

        tranval[condition1] = 1 + largebins

        tranval[condition2] = bincount

        tranval[condition4] = -1 + largebins

        tranval[condition5] = 0

        tranval[condition7] = -1 + largebins

    return tranval

# Transforms the Data that has a Positive Range
def dfzero2pos(data, alpha, datmax):
    alphas = []
    commonratios = []
    for col in data.columns:
        datacol = data[col].to_numpy()
        condition1 = np.where(datacol < alpha[col])
        condition2 = np.where(datacol >= alpha[col])

        alphas.append(alpha[col])
        commonratio = (datmax[col]/alpha[col])**(1/(bincount - 2))
        commonratios.append([commonratio])

        logged = np.log(datacol[condition2]/alpha[col])/np.log(commonratio)
        floored = np.floor(logged + 2)
        floated = np.float64(floored)

        datacol[condition2] = floated
        datacol[condition1] = 1
    
    return alphas, commonratios, data

def zero2pos(data, length, commonratio, datmin, datmax):
    data = data.to_numpy()
    condition1 = np.where(data < datmin)
    condition2 = np.where(data >= datmin)
    
    logged = np.log(data[condition2]/datmin)/np.log(commonratio)
    floored = np.floor(logged + 2)
    floated = np.float64(floored)

    data[condition2] = floated
    data[condition1] = 1
    
    return data
def dfneg2zero(data, datmin, alpha):
    alphas = []
    commonratios = []

    for col in data.columns:
        datacol = data[col].to_numpy()
        condition1 = np.where(datacol > alpha)
        condition2 = np.where(datacol <= alpha)

        alphas.append(alpha[col])
        commonratio = (datmin[col]/alpha[col])**(1/(bincount - 2))
        commonratios.append([commonratio])

        logged = np.log(datacol[condition2]/alpha[col])/np.log(commonratio)
        floored = np.floor(logged + 2) 
        floated = -1*np.float64(floored) + bincount   

        datacol[condition2] = floated
        datacol[condition1] = -1 + bincount  
    return alphas, commonratios, data

def neg2zero(data, length, commonratio, datmin, datmax): 
    data = data.to_numpy()
    condition1 = np.where(data > datmax)
    condition2 = np.where(data <= datmax)

    logged = np.log(data[condition2]/datmax)/np.log(commonratio)
    floored = np.floor(logged + 2) 
    floated = -1*np.float64(floored) + bincount

    data[condition2] = floated
    data[condition1] = -1 + bincount

    return data

def root(root, value):
    return round(value**(1/root), 6)

""" 1. Loading and binning data """
def is_number(value):
    try:
        float(value)
        return True
    except:
        return False

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


    linear_index = []
    yaxis = [0]
    column_bin_sizes = []
    quartile25to75 = []
    Datapoints = []
    Ind2 = []
    Ind2zero = []
    Position = []
    alphavals = []
    
    bin_stats = {'size': data.shape,
                 'min': data.min(),
                 'max': data.max(),
                 'twenty5': data.quantile(0.25),
                 'seventy5': data.quantile(0.75),
                 'alpha': (2*(data.quantile(0.75) - data.quantile(0.25)))/(data.shape[0]**(1/3))}

    nfeats = bin_stats['size'][1] 
    datalen = bin_stats['size'][0]
   
    positiverange = np.where((data.min() >= 0) & (data.max() > 0))[0]
    negativerange = np.where((data.min() < 0) & (data.max() <= 0))[0]
    neg2posrange =  np.where((data.min() < 0) & (data.max() > 0))[0]
    zeroalpha = np.where((2*(data.quantile(0.75) - data.quantile(0.25)))/(data.shape[0]**(1/3)) == 0)[0]

    positivedf = data.iloc[:, positiverange]
    negativedf = data.iloc[:, negativerange]
    neg2posdf = data.iloc[:, neg2posrange]
    zerodf = data.iloc[:, zeroalpha]


    positivenames = positivedf.columns
    negativenames = negativedf.columns
    neg2posnames = neg2posdf.columns
    zeronames = zerodf.columns 
    
    alphaspos, commonratiospos, positivedf = dfzero2pos(positivedf, bin_stats['alpha'][positivenames], bin_stats['max'][positivenames])
    yaxis = yaxis * len(positivenames)
    alphavals = alphaspos
    column_bin_sizes = commonratiospos
    positivedf.reset_index(drop = True, inplace = True)

    alphasneg, commonratiosneg, negativedf = dfneg2zero(negativedf, -1*bin_stats['alpha'][positivenames], bin_stats['min'][negativenames])
    yaxis = yaxis + ([bincount] * len(negativenames))
    alphavals = alphavals + alphasneg
    column_bin_sizes = column_bin_sizes + commonratiosneg
    negativedf.reset_index(drop = True, inplace = True)
    
    yvalues, alphasneg2pos, commonratiosneg2pos, neg2posdf = dfneg2pos(neg2posdf, bin_stats['alpha'][neg2posnames], bin_stats['min'][neg2posnames], bin_stats['max'][neg2posnames])
    yaxis = yaxis + yvalues
    alphavals = alphavals + alphasneg2pos
    column_bin_sizes = column_bin_sizes + commonratiosneg2pos
    neg2posdf.reset_index(drop = True, inplace = True)

    # zerodf = pandas.DataFrame(bincount + 55, index= np.arange(datalen), columns = zeronames)
    data = pandas.concat([positivedf, negativedf, neg2posdf], axis=1)
    column_names = data.columns

    bin_stats = {'size': data.shape,
                 'min': data.min(),
                 'max': data.max()}

    nfeats = bin_stats['size'][1] 
    datalen = bin_stats['size'][0]

    # for i in range(0, nfeats):
    #     quartile25to75.append(bin_stats['seventy5'][i] - bin_stats['twenty5'][i])
    #     alpha = firstAterm(quartile25to75[-1], datalen)
    #     alphavals.append(alpha)
    #     if alpha == 0:
    #         empty = [bincount + 55] * datalen
    #         data[column_names[i]] = pandas.Series(np.array(empty), index = range(0, datalen))
    #         column_bin_sizes.append([0])
    #         yaxis.append(0)
    #         data.drop(column_names[i], axis = 1, inplace = True)
    #         nfeats = nfeats - 1
    #         continue 
    #     if bin_stats['min'][i] >= 0 and bin_stats['max'][i] > 0:
    #         column_bin_sizes.append([root(bincount - 2, bin_stats['max'][i]/alpha)])
    #         data[column_names[i]] = zero2pos(data[column_names[i]], datalen, column_bin_sizes[-1][0], alpha, bin_stats['max'][i])
    #         yaxis.append(0)
    #     if bin_stats['min'][i] < 0 and bin_stats['max'][i] <= 0:
    #         alpha = alpha * -1
    #         column_bin_sizes.append([root(bincount - 2, bin_stats['min'][i]/(alpha))])
    #         yaxis.append(bincount)
    #         data[column_names[i]] = neg2zero(data[column_names[i]], datalen, column_bin_sizes[-1][0], bin_stats['min'][i], alpha)
    #     if bin_stats['min'][i] < 0 and bin_stats['max'][i] > 0:
    #         small = 0
    #         large = 0
    #         if abs(bin_stats['max'][i]) < abs(bin_stats['min'][i]):
    #             small = abs(bin_stats['max'][i])
    #             large = abs(bin_stats['min'][i])
    #         else:
    #             small = abs(bin_stats['min'][i])
    #             large = abs(bin_stats['max'][i])

    #         ratio = root(bincount + 1, (small*small)/(alpha*large))
    #         binssmall = int(math.floor(abs(1 + math.log(small/alpha, ratio))))
    #         binslarge = int(math.floor(abs(1 + math.log(large/alpha, ratio))))
    #         zfactor = binssmall/binslarge
    #         binssmall = round(bincount/(2 + 1/zfactor), 0)
    #         binslarge = binssmall + (bincount - (2*binssmall))
    #         small_interval = root(binssmall, small/alpha)
    #         large_interval = root(binslarge, large/alpha)
    #         column_bin_sizes.append([small_interval, large_interval])        
    #         if abs(bin_stats['max'][i]) > abs(bin_stats['min'][i]):
    #             yaxis.append(binssmall)
    #         else:
    #             yaxis.append(binslarge)
    #         data[column_names[i]] = neg2pos(data[column_names[i]], column_bin_sizes[-1], datalen, ratio, bin_stats['min'][i], bin_stats['max'][i], alpha, binssmall, binslarge)
        

    data_ind = pandas.notnull(data)  # Handle NaN values
    data[~data_ind] = bincount + 55          # Handle NaN values
    data = data.astype(np.uint16) # cast to save memory
    data[data==bincount] = bincount - 1         # in case of numerical precision issues



    # for i in range(0, nfeats):
    #     count = 0
    #     print("Column Name: ", column_names[i])
    #     print("MINIMUM TO MAXIMUM RANGE: ", bin_stats['min'][i], bin_stats['max'][i])
    #     print("Column Bin Sizes: ", column_bin_sizes[i])
    #     print("Bin Min and Max: ", bin_stats_transformed['min'][i], bin_stats_transformed['max'][i])
    #     print("Y Axis: ", yaxis[i])
    #     totalbins = bin_stats_transformed['max'][i] - bin_stats_transformed['min'][i]
    #     print("Range of Bins: ", totalbins)
    #     print("\n")
    #     for item in data[column_names[i]]:
    #         print(column_names[i], count, item)
    #         count = count + 1
    
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

    # Create a linear index for feature bins
  
    
    
    for feat1 in range(nfeats):
        name1 = column_names[feat1]
        feat1_tf = data[name1] * bincount

        for feat2 in range(feat1 + 1, nfeats):
            Datapoints.append([])
            Ind2.append([])
            Ind2zero.append([])
            Position.append([])
            name2 = column_names[feat2]
            feat2_tf = data[name2]
            feat1_tf = feat1_tf[data_ind[name1] & data_ind[name2]]
            feat2_tf = feat2_tf[data_ind[name1] & data_ind[name2]]
                      
            if feat2_tf.size<=1:
                continue
            
            # sort linear matrix indices
            SortedFeats = np.sort(feat1_tf + feat2_tf)
            # Do math to get the indices
            ind2 = np.diff(SortedFeats)                       
            ind2 = np.nonzero(ind2)[0]                       # nonzeros are cumulative sum of all bin values
            ind2 = np.append(ind2,SortedFeats.size-1)
            # print(feat2_sort.shape)
            rows = (SortedFeats[ind2]/bincount).astype(np.uint8)   # calculate row from linear index
            cols = np.mod(SortedFeats[ind2],bincount)              # calculate column from linear index
            counts = np.diff(ind2)                           # calculate the number of values in each bin
            bins[feat1,feat2,rows[0],cols[0]] = ind2[0] + 1
            bins[feat1,feat2,rows[1:],cols[1:]] = counts
            linear_index.append([feat1,feat2])
            

            # print(feat1_tf, feat2_tf)
            # print(" \n")
            # for val in range(0, datalen):
            #     #print((feat1_tf[val], feat2_tf[val]))
            #     Datapoints[-1].append((feat1_tf[val], feat2_tf[val]))
            # Datapoints[-1] = sorted(Datapoints[-1], key=lambda element: (element[0], element[1]))
            # for item in range(0, datalen - 1):
            #     #print(Datapoints[-1][item], Datapoints[-1][item+1], Datapoints[-1][item][0] - Datapoints[-1][item+1][0], Datapoints[-1][item][1] - Datapoints[-1][item+1][1])
            #     Ind2[-1].append([Datapoints[-1][item][0] - Datapoints[-1][item+1][0], Datapoints[-1][item][1] - Datapoints[-1][item+1][1]])
            #     if Ind2[-1][-1] != [0, 0]:
            #         Ind2zero[-1].append(Ind2[-1][-1])
            #         Position[-1].append(item)
            # UniqueData = list(set(Datapoints[-1]))
            # UniqueData = sorted(UniqueData, key = lambda element: (element[0], element[1]))

            # print("Length of No Zeros List: ", len(Ind2zero[-1]), Ind2zero[-1])
            # print("Length of Position List: ", len(Position[-1]), Position[-1])
            # print("Length of Unique Items: ", len(UniqueData), UniqueData)            
            # print("Length of Indexes: ", len(Ind2[-1]), Ind2[-1])
            # print("Length of Data: ", len(Datapoints[-1]), Datapoints[-1])
            
            # for i in range(0, len(UniqueData)):
            #     if i == 0:
            #         print("There are", str(Position[-1][i] + 1), "observations of", str(UniqueData[i]))
            #         #bins[feat1, feat2, UniqueData[i][0], UniqueData[i][1]] = Position[-1][i] + 1
            #     elif (i == (len(UniqueData)-1)):
            #         print("There are", str(datalen - Position[-1][i - 1]), "observations of", str(UniqueData[i]))
            #         #bins[feat1, feat2, UniqueData[i][0], UniqueData[i][1]] = datalen - Position[-1][i-1]
            #     else:
            #         print("There are", str(Position[-1][i] - Position[-1][i - 1]), "observations of", str(UniqueData[i]))    
            #         #bins[feat1, feat2, UniqueData[i][0], UniqueData[i][1]] = Position[-1][i] - Position[-1][i - 1]
            # print(name1, name2, "NEXT")
            # print(" ")
            
    return yaxis, bins, bin_stats, linear_index, column_bin_sizes, alphavals

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
        bins - A numpy matrix that has shape (nfeats,nfeats,200,200)
        bin_feats - A list containing the minimum and maximum values of each column
        linear_index - Numeric value of column index from original csv
    """

    # Get basic column statistics and bin sizes
    nfeats = len(column_names)
    yaxis = np.zeros(nfeats, dtype=int)
    alphavals = yaxis
    bin_stats = {'min': data.min(),
                 'max': data.max()}
    column_bin_size = (bin_stats['max'] * (1 + 10**-6) - bin_stats['min'])/bincount

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

    # Create a linear index for feature bins
    linear_index = []

    # Bin the data
    for feat1 in range(nfeats):
        if bin_stats['min'][feat1] >= 0:
            yaxis[feat1] = 0
        else:
            yaxis[feat1] = abs(bin_stats['min'][feat1])/column_bin_size[feat1] + 1
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
    return yaxis, bins, bin_stats, linear_index, column_bin_size, alphavals

""" 2. Plot Generation """
def format_ticks_log(fmin,fmax,nticks, yaxis, commonratio, alphavalue):
    """ Generate tick labels
    Polus Plots uses D3 to generate the plots. This function tries to mimic
    the formatting of tick labels. Tick labels have a fixed width, and in
    place of using scientific notation a scale prefix is appen ded to the end
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
    
    out = [(alphavalue*(commonratio[-1]**(t-yaxis))) if yaxis<t else (-1*(alphavalue*(commonratio[0]**(yaxis-t))) if yaxis>t else 0) for t in np.arange(fmin,fmax,(fmax-fmin)/(nticks-1))]
    if yaxis < fmax:
        out.append(alphavalue*(commonratio[-1]**(fmax-yaxis)))
    elif yaxis > fmax:
        out.append(-1*(alphavalue*(commonratio[0]**(yaxis-fmax))))
    else:
        out.append(0)
    
    fticks = []
    convertprefix = []

    for i in range(nticks):
        formtick = "%#.3f" % out[i]
        decformtick = '%.2e' % Decimal(formtick)
        #print(i, ")", formtick, decformtick)
        convertexponent = int(decformtick[-3:])
        numbers = float(decformtick[:-4])
        if convertexponent > 0:
            if convertexponent % 3 == 2:
                movednum = round(numbers/10,2)
                newprefix = _prefix[int(convertexponent + 1)]
                formtick = str(movednum) + newprefix
            elif convertexponent % 3 == 1:
                movednum = round(numbers*10,1)
                newprefix = _prefix[int(convertexponent - 1)]
                formtick = str(movednum) + newprefix
            else:
                newprefix = _prefix[convertexponent]
                if out[i] < 0:
                    formtick = str(decformtick[:5]) + newprefix
                else: 
                    formtick = str(decformtick[:4]) + newprefix
        elif convertexponent < 0:
            if convertexponent % -3 == -2:
                movednum = round(numbers*10,1)
                newprefix = _prefix[int(convertexponent - 1)]
                formtick = str(movednum) + newprefix
            elif convertexponent % -3 == -1:
                movednum = round(numbers/10,2)
                newprefix = _prefix[int(convertexponent + 1)]
                formtick = str(movednum) + newprefix
            else:
                newprefix = _prefix[convertexponent]
                if out[i] < 0:
                    formtick = str(decformtick[:5]) + newprefix
                else: 
                    formtick = str(decformtick[:4]) + newprefix
        else:
            if out[i] < 0:
                formtick = str(decformtick[:5]) + _prefix[convertexponent]
            else: 
                formtick = str(decformtick[:4]) + _prefix[convertexponent]
        convertprefix.append(convertexponent)
        fticks.append(formtick)

    return fticks
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

    fticks = []
    convertprefix = []
    for i in range(nticks):
        formtick = "%#.3f" % out[i]
        decformtick = '%.2e' % Decimal(formtick)
        #print(i, ")", formtick, decformtick)
        convertexponent = int(decformtick[-3:])
        numbers = float(decformtick[:-4])
        if convertexponent > 0:
            if convertexponent % 3 == 2:
                movednum = round(numbers/10,2)
                newprefix = _prefix[int(convertexponent + 1)]
                formtick = str(movednum) + newprefix
            elif convertexponent % 3 == 1:
                movednum = round(numbers*10,1)
                newprefix = _prefix[int(convertexponent - 1)]
                formtick = str(movednum) + newprefix
            else:
                newprefix = _prefix[convertexponent]
                if out[i] < 0:
                    formtick = str(decformtick[:5]) + newprefix
                else: 
                    formtick = str(decformtick[:4]) + newprefix
        elif convertexponent < 0:
            if convertexponent % -3 == -2:
                movednum = round(numbers*10,1)
                newprefix = _prefix[int(convertexponent - 1)]
                formtick = str(movednum) + newprefix
            elif convertexponent % -3 == -1:
                movednum = round(numbers/10,2)
                newprefix = _prefix[int(convertexponent + 1)]
                formtick = str(movednum) + newprefix
            else:
                newprefix = _prefix[convertexponent]
                if out[i] < 0:
                    formtick = str(decformtick[:5]) + newprefix
                else: 
                    formtick = str(decformtick[:4]) + newprefix
        else:
            if out[i] < 0:
                formtick = str(decformtick[:5]) + _prefix[convertexponent]
            else: 
                formtick = str(decformtick[:4]) + _prefix[convertexponent]
        convertprefix.append(convertexponent)
        fticks.append(formtick)

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
             alphavals,
             bins,
             binsizes,
             column_names,
             bin_stats,
             fig,
             ax,
             data,
             axiszero,
             typegraph):
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

    #print("Row, Column:", r, c)
    data.set_data(np.ceil(d/d.max() *255))
    data.set_clim(0, 255)


    sizefont = 12 
    axlabel = fig.axes[1]
    aylabel = fig.axes[2]

    if len(axlabel.texts) == 0:
        axlabel.text(0.5, 0.5, "\n".join(wrap(column_names[c], 60)), va = 'center', ha = 'center', fontsize = sizefont, wrap = True)
        bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = sizefont - 1
        while (bbxtext.x0 < 0 or bbxtext.x1 > CHUNK_SIZE) or (bbxtext.y0 < 0 or bbxtext.y1 > CHUNK_SIZE*.075):
            axlabel.texts[0].set_fontsize(decreasefont)
            bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
            decreasefont = decreasefont - 1 
        #print("X Label Size: ", bbxtext)
    else:
        axlabel.texts[0].set_text("\n".join(wrap(column_names[c], 60)))
        axlabel.texts[0].set_fontsize(sizefont)
        bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = sizefont - 1
        while (bbxtext.x0 < 0 or bbxtext.x1 > CHUNK_SIZE) or (bbxtext.y0 < 0 or bbxtext.y1 > (CHUNK_SIZE*.075)):
            axlabel.texts[0].set_fontsize(decreasefont)
            bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
            decreasefont = decreasefont - 1 
        #print("X Label Size: ", bbxtext)

    if len(aylabel.texts) == 0:
        aylabel.text(0.5, 0.5, "\n".join(wrap(column_names[r], 60)), va = 'center', ha = 'center', fontsize = sizefont, rotation = 90, wrap = True)
        bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = sizefont - 1
        while (bbytext.y0 < 0 or bbytext.y1 > CHUNK_SIZE) or (bbytext.x0 < 0 or bbytext.x1 > (CHUNK_SIZE*.075)):
            aylabel.texts[0].set_fontsize(decreasefont)
            bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
            decreasefont = decreasefont - 1 
        #print("X Label Size: ", bbytext)
    else:
        aylabel.texts[0].set_text("\n".join(wrap(column_names[r], 60)))
        aylabel.texts[0].set_fontsize(sizefont)
        bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = sizefont - 1
        while (bbytext.y0 < 0 or bbytext.y1 > CHUNK_SIZE) or (bbytext.x0 < 0 or bbytext.x1 > (CHUNK_SIZE*.075)):
            aylabel.texts[0].set_fontsize(decreasefont)
            bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
            decreasefont = decreasefont - 1 
        #print("X Label Size: ", bbytext)
    
    if typegraph == "linear":
        #print("Data Column: ", c)
        ax.set_xticklabels(format_ticks(bin_stats['min'][column_names[c]],bin_stats['max'][column_names[c]],11),
                        rotation=45, fontsize = 5, ha='right')
        #print("Data Column: ", r)
        ax.set_yticklabels(format_ticks(bin_stats['min'][column_names[r]],bin_stats['max'][column_names[r]],11), 
                        fontsize = 5, ha='right')
    if typegraph == "log":
        #print("Data Column: ", c)
        ax.set_xticklabels(format_ticks_log(0,bincount,11, axiszero[c], binsizes[c], alphavals[c]),
                        rotation=45, fontsize = 5, ha='right')
        #print("Data Column: ", r)
        ax.set_yticklabels(format_ticks_log(0,bincount,11, axiszero[r], binsizes[r], alphavals[r]),
                        fontsize = 5, ha='right')

    if len(ax.lines) == 0:
        if axiszero[c] == 0:
            ax.axvline(x=axiszero[c])
        else:
            ax.axvline(x=axiszero[c] + 0.5)
        if axiszero[r] == 0:
            ax.axhline(y=axiszero[r])
        else:
            ax.axhline(y=axiszero[r] + 0.5)
    else:
        ax.lines[-1].remove()
        ax.lines[-1].remove()
        if axiszero[c] == 0:
            ax.axvline(x=axiszero[c])
        else:
            ax.axvline(x=axiszero[c] + 0.5)
        if axiszero[r] == 0:
            ax.axhline(y=axiszero[r])
        else:
            ax.axhline(y=axiszero[r] + 0.5)
    
    # textlist = []
    # if len(ax.texts) == 0:
    #     for i in range(0, bincount):  
    #         for j in range(0, bincount):
    #             textingraph = ax.text(j + 0.5, i + 0.5, d[i,j], ha="center", va = "center", fontsize = 2.5)
    #             textlist.append([i, j])
    # else:
    #     for txt in ax.texts:
    #         print(txt, type(txt))
    #         pos = str(txt)[5:-1]
    #         pos = pos.split(",")
    #         i = float(pos[1])
    #         j = float(pos[0])
    #         txt.set_text(d[int(i - 0.5),int(j - 0.5)])

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
    #fig, ax = plt.subplots(dpi=int(CHUNK_SIZE/4),figsize=(4,4),constrained_layout = True)
    #plt.rcParams['figure.constrained_layout.use'] = True
    #plt.rcParams['figure.figsize'] = 4.5, 4
    #datacolor = ax.pcolorfast(np.zeros((CHUNK_SIZE,CHUNK_SIZE),np.uint64),cmap=cmap)
    datacolor = ax.pcolorfast(np.zeros((bincount, bincount),np.uint64),cmap=cmap)
    ticks = [(t + 0.5) for t in range(0, bincount +1, int(bincount/10))]
 
    #ticks = np.logspace(1, bincount, num = bincount/10)
    #ticks = ticks.tolist()

    #ticks.append(bincount - 1)

    #leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad = 0)

    ax.set_xlim(0,bincount)
    ax.set_ylim(0,bincount)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    


    ax.set_xticklabels(ticks, rotation = 45)
    ax.set_yticklabels(ticks)

    fig.canvas.draw()

    axlabel = fig.add_axes([.075, 0, 1, .075], frameon = False, alpha = .5, facecolor = 'b')
    axlabel.set_xticks([])
    axlabel.set_yticks([])
    axlabel.set_clip_on(True)
    aylabel = fig.add_axes([0, .075, .075, 1], frameon = False, alpha = .5, facecolor = 'b')
    aylabel.set_xticks([])
    aylabel.set_yticks([])
    aylabel.set_clip_on(True)
    #axlabel.text(0.5, 0.5, '${Needs X Axes Label}$', ha = 'center', va = 'center')
    #aylabel.text(0.5, 0.5, '${Needs Y Axes Label}$', ha = 'center', va = 'center', rotation = 90)
    fig.add_axes(axlabel, aylabel)
    return fig, ax, datacolor

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

def metadata_to_graph_info(bins,outPath,outFile, indexscale):
    
    # Create an output path object for the info file
    op = Path(outPath).joinpath("{}.dzi".format(outFile))
    
    # create an output path for the images
    of = Path(outPath).joinpath('{}_files'.format(outFile))
    of.mkdir(exist_ok=True)
    
    # Get metadata info from the bfio reader
    ngraphs = len(indexscale) #(= NumberofColumns Choose 2)
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
def _get_higher_res(typegraph, S,info,cnames, outpath,out_file,indexscale,bintype,binstats, binsizes, axiszero, alphavals, X=None,Y=None):
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
        if index>=len(indexscale):
            image = np.ones((CHUNK_SIZE,CHUNK_SIZE,4),dtype=np.uint8) * (bincount + 55)
        else:
            image = gen_plot(indexscale[index][0],
                             indexscale[index][1],
                             alphavals,
                             bintype,
                             binsizes,
                             cnames,
                             binstats,
                             fig,
                             ax,
                             datacolor,
                             axiszero,
                             typegraph)
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
                    sub_image = _get_higher_res_par(typegraph,
                                                    S+1,
                                                   info,
                                                   cnames,
                                                   outpath,
                                                   out_file,
                                                   indexscale,
                                                   bintype,
                                                   binstats,
                                                   binsizes,
                                                   axiszero,
                                                   alphavals,
                                                   X=subgrid_dims[0][x:x+2],
                                                   Y=subgrid_dims[1][y:y+2])
                else:
                    sub_image = _get_higher_res(typegraph,
                                                S+1,
                                               info,
                                               cnames,
                                               outpath,
                                               out_file,
                                               indexscale,
                                               bintype,
                                               binstats,
                                               binsizes,
                                               axiszero,
                                               alphavals,
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
def _get_higher_res_par(typegraph, S,info, cnames, outpath,out_file,indexscale, bintype, binstats, binsizes, axiszero, alphavals, X=None,Y=None):
    # Get the scale info
    processID = os.getpid()
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
        if index>=len(indexscale):
            image = np.ones((CHUNK_SIZE,CHUNK_SIZE,4),dtype=np.uint8) * (bincount + 55)
        else:
            image = gen_plot(indexscale[index][0],
                             indexscale[index][1],
                             alphavals,
                             bintype,
                             binsizes,
                             cnames,
                             binstats,
                             fig,
                             ax,
                             datacolor,
                             axiszero,
                             typegraph)
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
                    subgrid_images.append(pool.apply_async(_get_higher_res,(typegraph, 
                                                                            S+1,
                                                                           info,
                                                                           cnames,
                                                                           outpath,
                                                                           out_file,
                                                                           indexscale,
                                                                           bintype,
                                                                           binstats,
                                                                           binsizes,
                                                                           axiszero,
                                                                           alphavals,
                                                                           subgrid_dims[0][x:x+2],
                                                                           subgrid_dims[1][y:y+2])))
                    
            for y in range(0,len(subgrid_dims[1])-1):
                y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
                y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
                for x in range(0,len(subgrid_dims[0])-1):
                    x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                    x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                    image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],:] = _avg2(subgrid_images[y*(len(subgrid_dims[0])-1) + x].get())
        # print("Subgrid Images")
        # print(subgrid_images)

        # for item in subgrid_images:
        #     print(item.release())
        # file_grid = open("data_subgrid_images.pkl", 'wb')
        # # del subgrid_images['lock']
        # pickle.dump(subgrid_images, file_grid)
        # file_grid.close()

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
                        required=True,
                        )

    parser.add_argument('--outDirLinear',               # Pyramid directory
                        dest='outDirLinear',
                        type=str,
                        help='The output directory for the flatfield images of Linear Scaled Graphs.',
                        required=True)

    parser.add_argument('--outDirLog',
                        dest='outDirLog',
                        type=str,
                        help='The output directory for the flatfield images of Log Scaled Graphs.',
                        required=True)
    
    """ Get the input arguments """
    args = parser.parse_args()

    input_path = args.inpDir
    linear_output_path = Path(args.outDirLinear)
    log_output_path = Path(args.outDirLog)

    # input_path = "/Users/mmvihani/AxleInformatics/polus-plugins/polus-graph-pyramid-builder-plugin/input/"
    # linear_output_path = "/Users/mmvihani/AxleInformatics/polus-plugins/polus-graph-pyramid-builder-plugin/outDirLinear/"
    # log_output_path = "/Users/mmvihani/AxleInformatics/polus-plugins/polus-graph-pyramid-builder-plugin/outDirLog/"

    logger.info('inpDir = {}'.format(input_path))
    logger.info('outDirLinear = {}'.format(linear_output_path))
    logger.info('outDirLog = {}'.format(log_output_path))

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
        data_log, cnames_log = load_csv(f)
        column_names = data.columns
        column_names_log = data_log.columns
        logger.info('Done loading csv!')

        # Bin the data
        logger.info('Binning data for {} features...'.format(column_names.size))
        start = time.time()
        yaxis_log, log_bins, log_bin_stats, log_index, log_binsizes, alphavals_log = bin_data_log(data_log, column_names_log)
        endlog = time.time()
        yaxis_linear, bins, bin_stats, linear_index, linear_binsizes, alphavals_linear = bin_data(data,column_names)
        endlinear = time.time()
        print("Binning Data for Log Graphs take: ", endlog - start)
        print("Binning Data for Linear Graphs take ", endlinear - endlog)
        print("Binning data for Log Scale graphs are ", (endlog - start)/(endlinear - endlog), "times slower.")
        logger.info('Done!')
        del data    # get rid of the original data to save memory
        del data_log

        # Generate the default figure components
        logger.info('Generating colormap and default figure...')
        cmap = get_cmap()
        fig, ax, datacolor = get_default_fig(cmap)
        logger.info('Done!')

        # Generate the dzi file
        logger.info('Generating pyramid metadata...')
        start = time.time()
        info_linear = metadata_to_graph_info(bins,linear_output_path,folder, linear_index)
        endlinear = time.time()
        info_log = metadata_to_graph_info(log_bins, log_output_path,folder, log_index)
        endlog = time.time()
        # print("INFO LINEAR")
        # print(info_linear)
        # print("INFO LOG")
        # print(info_log)
        print("Info for for Log Graphs take: ", endlog - endlinear)
        print("Info for Linear Graphs take ", endlinear - start)
        print("Info for Log Scale graphs are ", (endlog - endlinear)/(endlinear - start), "times slower.")

        logger.info('Done!')
        
        logger.info('Writing layout file...!')
        write_csv(cnames,linear_index,info_linear,linear_output_path,folder)  
        write_csv(cnames_log, log_index, info_log, log_output_path, folder)
        logger.info('Done!')

        # Create the pyramid
        logger.info('Building pyramid...')
        image_linear = _get_higher_res("linear", 0, info_linear,column_names, linear_output_path,folder,linear_index, bins, bin_stats, linear_binsizes, yaxis_linear, alphavals_linear)
        image_log = _get_higher_res("log", 0, info_log, column_names_log, log_output_path, folder, log_index,log_bins, log_bin_stats, log_binsizes, yaxis_log, alphavals_log)