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
bincount = 50 #MUST BE EVEN NUMBER

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
            bin = (-1.0) + bincount
            replacedata.append(bin)  
        else:
            log = math.log(val/max, commonratio)
            bin = float(math.floor(log + 2))
            replacedata.append((-1*bin) + bincount)
    return pandas.Series(np.array(replacedata), index = range(0, length))

def neg2pos(data, binsizes, length, commonratio, min, max, alpha, smallbins, largebins):
    replacedata = []
    # print("Number of bins on both sides in function: ", smallbins, largebins, smallbins + largebins)
    # print("Min and Max Data: ", min, max)
    # print("Alpha: ", alpha)
    # print("Commonratio: ", commonratio)
    # print("Binsizes (small, large): ", binsizes)
    binnumber = 0
    if abs(max) > abs(min):
        yaxis = smallbins
    else:
        yaxis = largebins
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
                replacedata.append(1.0 + yaxis)
            elif val == max:
                replacedata.append(binnumber + yaxis)
            else:
                log = math.log(val/alpha, commonratio)
                bin = float(math.floor(log + 1))
                replacedata.append(abs(bin) + yaxis) 
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
                replacedata.append(-1.0 + yaxis)
            elif val == min:
                replacedata.append((-1*binnumber) + yaxis)
            else:
                log = math.log(val/(-1*alpha), commonratio)
                bin = -1*float(math.floor(log + 1))
                replacedata.append(bin + yaxis)
                # if abs(max) > abs(min) and replacedata[-1] > smallbins:
                #     print("ERROR NEG Large: ", val, replacedata[-1])
                # if abs(max) < abs(min) and replacedata[-1] > largebins:
                #     print("ERROR NEG Small: ", val, replacedata[-1])
    return yaxis, pandas.Series(np.array(replacedata), index = range(0, length))  
                 

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
        # print(" ")
        # print("Difference: ", diff)
        # print("Range Edited", Range)
        # print("Edited Transformation: ", transformed)
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
    yaxis = []
    column_bin_sizes = []
    quartile25to75 = []
    Datapoints = []
    Ind2 = []
    Ind2zero = []
    Position = []
    
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
            column_bin_sizes.append([0])
            yaxis.append(0)
            continue 
        if bin_stats['min'][i] >= 0 and bin_stats['max'][i] > 0:
            column_bin_sizes.append([root(bincount - 2, bin_stats['max'][i]/alpha)])
            data[column_names[i]] = zero2pos(data[column_names[i]], datalen, column_bin_sizes[-1][0], alpha, bin_stats['max'][i])
            yaxis.append(0)
        if bin_stats['min'][i] < 0 and bin_stats['max'][i] <= 0:
            alpha = alpha * -1
            column_bin_sizes.append([root(bincount - 2, bin_stats['min'][i]/(alpha))])
            yaxis.append(bincount)
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
            yval, data[column_names[i]] = neg2pos(data[column_names[i]], column_bin_sizes[-1], datalen, ratio, bin_stats['min'][i], bin_stats['max'][i], alpha, binssmall, binslarge)
            yaxis.append(yval)

    bin_stats_transformed = {'min': data.min(),
                             'max': data.max()}

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
        dtype = np.u=uint8
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
            
    return yaxis, bins, bin_stats, bin_stats_transformed, linear_index

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
    data.set_clim(np.min(d),np.max(d)) #color limit
    #data.set_cmap(cmap)

    # count = 0 
    # print(column_names[c], column_names[r])
    # for item in d:
    #     print(count, item)
    #     count = count + 1
    
    # print(" ")

    # ax.set_xlabel(column_names[c])
    # ax.set_xticklabels(format_ticks(bin_stats['min'][column_names[c]],bin_stats['max'][column_names[c]],int(bincount/10 + 1)),
    #                    rotation=45)
    # ax.set_ylabel(column_names[r])
    # ax.set_yticklabels(format_ticks(bin_stats['min'][column_names[r]],bin_stats['max'][column_names[r]],int(bincount/10 + 1)))

    ax.set_xlabel(column_names[c])
    ax.set_xticklabels(format_ticks(0,bincount,11),
                       rotation=45)
    ax.set_ylabel(column_names[r])
    ax.set_yticklabels(format_ticks(0,bincount,11))
    
    # textlist = []
    # if len(ax.texts) == 0:
    #     for i in range(0, bincount):
    #         for j in range(0, bincount):
    #             textingraph = ax.text(j, i, d[i,j], ha="center", va = "center", fontsize = 2.5)
    #             textlist.append([i, j])
    # else:
    #     for txt in ax.texts:
    #         pos = str(txt)[5:-1]
    #         pos = pos.split(",")
    #         i = int(pos[1])
    #         j = int(pos[0])
    #         txt.set_text(d[i,j])
            # print(txt, type(txt), pos, i, j)

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
    #datacolor = ax.pcolorfast(np.zeros((CHUNK_SIZE,CHUNK_SIZE),np.uint64),cmap=cmap)
    datacolor = ax.pcolorfast(np.zeros((bincount, bincount),np.uint64),cmap=cmap)
    ticks = [t for t in range(0, bincount +1, int(bincount/10))]
 
    #ticks = np.logspace(1, bincount, num = bincount/10)
    #ticks = ticks.tolist()

    #ticks.append(bincount - 1)
    ax.set_xlim(0,bincount)
    ax.set_ylim(0,bincount)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    ax.set_xticklabels(ticks, rotation = 45)
    ax.set_yticklabels(ticks)
    # ax.set_xticklabels(["     ", 
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     "],
    #                 rotation=45)
    # ax.set_ylabel(" ")
    # ax.set_yticklabels(["     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     ",
    #                     "     "])
    fig.canvas.draw()
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
def _get_higher_res(S,info,outpath,out_file,indexscale,bintype,binstats,X=None,Y=None,):

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
                             bintype,
                             column_names_log,
                             binstats,
                             fig,
                             ax,
                             datacolor)
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
                                                   indexscale,
                                                   bintype,
                                                   binstats,
                                                   X=subgrid_dims[0][x:x+2],
                                                   Y=subgrid_dims[1][y:y+2])
                else:
                    sub_image = _get_higher_res(S+1,
                                               info,
                                               outpath,
                                               out_file,
                                               indexscale,
                                               bintype,
                                               binstats,
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
def _get_higher_res_par(S,info,outpath,out_file,indexscale, bintype, binstats, X=None,Y=None):
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
        if index>=len(indexscale):
            image = np.ones((CHUNK_SIZE,CHUNK_SIZE,4),dtype=np.uint8) * (bincount + 55)
        else:
            image = gen_plot(indexscale[index][0],
                             indexscale[index][1],
                             bintype,
                             column_names_log,
                             binstats,
                             fig,
                             ax,
                             datacolor)
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
                                                                           indexscale,
                                                                           bintype,
                                                                           binstats,
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
        yaxis, log_bins, log_bin_stats, log_bin_trans, log_index = bin_data_log(data_log, column_names_log)
        bins, bin_stats, linear_index = bin_data(data,column_names)

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
        info_linear = metadata_to_graph_info(bins,linear_output_path,folder, linear_index)
        info_log = metadata_to_graph_info(log_bins, log_output_path,folder, log_index)
        logger.info('Done!')
        
        logger.info('Writing layout file...!')
        write_csv(cnames,linear_index,info_linear,linear_output_path,folder)  
        write_csv(cnames_log, log_index, info_log, log_output_path, folder)
        logger.info('Done!')

        # Create the pyramid
        logger.info('Building pyramid...')
        image_linear = _get_higher_res(0, info_linear,linear_output_path,folder,linear_index, bins, bin_stats)
        image_log = _get_higher_res(0, info_log, log_output_path, folder, log_index,log_bins, log_bin_stats)