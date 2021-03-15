import pandas, multiprocessing, argparse, logging, matplotlib, copy, imageio
matplotlib.use('agg')
import numpy as np
from multiprocessing import Pool
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from pathlib import Path
import sys 
import math
import decimal
from decimal import Decimal
from textwrap import wrap
import os
import time

# Chunk Scale
CHUNK_SIZE = 1024

# Number of Ticks on the Axis Graph 
numticks = 11

# DZI file template
DZI = '<?xml version="1.0" encoding="utf-8"?><Image TileSize="' + str(CHUNK_SIZE) + '" Overlap="0" Format="png" xmlns="http://schemas.microsoft.com/deepzoom/2008"><Size Width="{}" Height="{}"/></Image>'

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

""" 1. Loading and binning data """
def is_number(value):
    """ This function checks to see if the value can be converted to a number """
    try:
        float(value)
        return True
    except:
        return False

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

def bin_data(data, yaxis, typegraph, column_bin_size, bin_stats):
    """ This function bins the data """

    column_names = data.columns
    nfeats = bin_stats['size'][1] 
    datalen = bin_stats['size'][0]

    data_ind = pandas.notnull(data)  # Handle NaN values
    data[~data_ind] = 255          # Handle NaN values
    data = data.astype(np.uint16) # cast to save memory
    data[data>=bincount] = bincount - 1 # in case of numerical precision issues

    nrows = data.shape[0]
    if nrows < 2**8:
        dtype = np.uint8
    elif nrows < 2**16:
        dtype = np.uint16
    elif nrows < 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    
    
    totalgraphs = int((nfeats**2 - nfeats)/2)
    bins = np.zeros((totalgraphs, bincount, bincount), dtype=dtype)
    graph_index = []
    graph_dict = {}

    # Create a linear index for feature bins
    i = 0
    for feat1 in range(nfeats):
        name1 = column_names[feat1]
        feat1_tf = data[name1] * bincount
        if typegraph == "linear":
            if bin_stats['min'][feat1] >= 0:
                yaxis[feat1] = 0
            else:
                yaxis[feat1] = abs(bin_stats['min'][feat1])/column_bin_size[feat1] + 1

        for feat2 in range(feat1 + 1, nfeats):
            graph_dict[(feat1, feat2)] = i
            
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
            rows = (SortedFeats[ind2]/bincount).astype(np.uint8)   # calculate row from linear index
            cols = np.mod(SortedFeats[ind2],bincount)              # calculate column from linear index
            counts = np.diff(ind2)                           # calculate the number of values in each bin
            bins[i,rows[0],cols[0]] = ind2[0] + 1 
            bins[i,rows[1:],cols[1:]] = counts 
            graph_index.append([feat1,feat2])
            i = i + 1

    return bins, bin_stats, graph_index, graph_dict, yaxis

def transform_data_log(data,column_names):
    """ Bin the data
    
    Data from a pandas Dataframe is binned in two dimensions in a logarithmic scale. 
    Binning is performed by binning data in one column along one axis and another column 
    is binned along theother axis. All combinations of columns are binned without repeats 
    or transposition. 

    To figure out which bin the data belongs in is solved by using the geometric sequence
    where:
    a(n) = a1*(r^(n-1))
        n = number of bins
        a1 = first number in range (scale factor)
        r = common ratio
        where r (bin width) is calculated by the Freedman Diaconis Rule: 
            https://en.wikipedia.org/wiki/Freedman–Diaconis_rule

    Every column of data falls into three different categories:
    1) dfzero2pos: All the numbers in the column are >= 0
    2) dfneg2zero: All the numbers in the column are <= 0
    3) dfneg2pos: The numbers in the column are either >=0 or <= 0 
                    The number of bins, alpha, and ratio, vary to the left and right of the
                    the y axis based on the range of the negative numbers and range of 
                    postive numbers, respectively.
            
    Inputs:
        data - A pandas Dataframe, with nfeats number of columns
        column_names - Names of Dataframe columns
    Outputs:
        bins - A numpy matrix that has shape (nfeats,nfeats,bincount,bincount)
        bin_feats - A list containing the minimum and maximum values of each column
        linear_index - Numeric value of column index from original csv
    """
    #Transforms the DataFrame that Range from a Negative Number to a Positive Number
    def dfneg2pos(data, alphavals, datmin, datmax, datalen):

        yaxis = [] 
        commonratios = []
        alphas = []

        # DETERMINING NUMBER OF BINS ON EACH SIDE OF Y AXIS

            # Need to figure out whether the negative range or 
            #   positive range is bigger
        minmax = pandas.concat([datmin, datmax], axis=1).abs()
        ispositivebigger = minmax.iloc[:,0] < minmax.iloc[:,1]
        smallside_an_values = minmax.min(axis=1) 
        largeside_an_values = minmax.max(axis=1)

        # NEED TO SOLVE FOR HOW MANY BINS THE SMALL RANGE AND LARGE RANGE REQUIRES
            # THREE VARIABLES ARE UNKNOWN
            # 1) the ratio 
            # 2) number of bins needed for smaller range
            # 3) number of bins needed for larger range

        # Solving for ratio (r), use equations:
            # x + y + 1= bincount
                # where x is the number of bins used for the smaller range
                # where y is the number of bins used for the larger range
                # where the +1 is for values that fall between -a1 and +a1 (-a1 is -alpha, +a1 is +alpha)
            # small = alpha*(r^(x-1)) --> log(r)[small/alpha] = x -1
            # large = alpha*(r^(y-1)) --> log(r)[large/alpha] = y - 1
            # Combining all equations we get the following: 
        ratios = ((smallside_an_values*largeside_an_values)/(alphavals*alphavals))**(1/(bincount-3))
        
        # use ratio to solve for number bins for both the large and small ranges
        num_bins_for_smallrange = np.round(abs(1 + np.log(smallside_an_values/alphavals)/np.log(ratios)))
        num_bins_for_largerange = np.round(abs(1 + np.log(largeside_an_values/alphavals)/np.log(ratios)))
            # sum of num_bins_for_smallrange + num_bins_for_largerange = bincount - 1

        small_lessthanalpha = smallside_an_values < alphavals
        large_lessthanalpha = largeside_an_values < alphavals

        num_bins_for_smallrange[small_lessthanalpha] = 0
        num_bins_for_largerange[large_lessthanalpha] = 0

        posbinslarge = num_bins_for_largerange[ispositivebigger.iloc[:]]
        posbinssmall = num_bins_for_smallrange[ispositivebigger.iloc[:] == False]
        posbins = posbinslarge.append(posbinssmall)
        

        negbinslarge = num_bins_for_largerange[ispositivebigger.iloc[:] == False]
        negbinssmall = num_bins_for_smallrange[ispositivebigger.iloc[:]]
        negbins = negbinslarge.append(negbinssmall)
        negbinsframe = negbins.to_frame().transpose()
        negbinsframe = negbinsframe.loc[negbinsframe.index.repeat(datalen)].reset_index(drop=True)

        ispositive = data.iloc[0:datalen,:] >= alphavals # Any positive value that is large than the first value of range
        isnegative = data.iloc[0:datalen,:] <= -1*alphavals # Any negative value that is smaller than the first value of range
        ismin = (data.iloc[0:datalen,:] == datmin) # Smallest value in column
        ismax = (data.iloc[0:datalen,:] == datmax) # Largest value in column

        binzero = np.log(np.abs(data.iloc[0:datalen,:])/alphavals)
        binned_data =  np.round(binzero/np.log(ratios)) # Calculate the bin for all values
        binned_data[binned_data==-0.0] = 0 # To replace -0.0 values with 0
        binned_data[binned_data==np.inf] = 0
        binned_data[binned_data==-np.inf] = 0 

        # Bins are 1-Indexed, and shifting all bin values to positive range.
                # Bin values are 0-Bincount
        data[binzero<=0] = negbinsframe[binzero<=0] + 1 # if -alpha < value < alpha, then it is negbin + 1
        data[ispositive] = (binned_data[ispositive] + negbinsframe[ispositive] + 1) 
        data[isnegative] = (negbinsframe[isnegative] - binned_data[isnegative] + 1)
        data[ismin] = 1 # Smallest bin value is 1.
        data[ismax] = bincount # Largest bin value is bincount
        data[data == -np.inf] = negbinsframe[data == -np.inf] + 1
        data[data == np.inf] = negbinsframe[data == np.inf] + 1

        commonratios = ratios.to_list()
        alphas = alphavals.to_list()
        yaxis = (negbins + 1).to_list()

        return yaxis, alphas, commonratios, data

    # Transforms the Data that has a Positive Range
    def dfzero2pos(data, alphavals, datmax, datalen):

        commonratios = (datmax/alphavals)**(1/(bincount - 1))
        
        binned_data = np.round(np.log(np.abs(data.iloc[0:datalen,:])/alphavals)/np.log(commonratios))
        binned_data[binned_data==-0.0] = 0 
        ismax = (data.iloc[0:datalen,:] == datmax)
        
        data[ismax] = bincount
        data[binned_data<=0] = 1
        data[binned_data>0] = binned_data[binned_data>0] + 1
        data[data == -np.inf] = 1
        data[data == np.inf] = 1

        alphas = alphavals.to_list()
        commonratios = commonratios.to_list()

        return alphas, commonratios, data

    # Transform the Data that has a Negative Range
    def dfneg2zero(data, datmin, alphavals, datalen):

        commonratios = (datmin/alphavals)**(1/(bincount - 1))
        
        binned_data =  np.round(np.log(data.iloc[0:datalen,:]/alphavals)/np.log(commonratios)) # Calculate the bin for all values
        binned_data[binned_data==-0.0] = 0 # To replace -0.0 values with 0
        ismin = (data.iloc[0:datalen,:] == datmin)

        data[ismin] =  1
        data[binned_data<=0] = bincount - 1
        data[binned_data>0] = bincount - binned_data[binned_data>0] + 1
        data[data == -np.inf] = bincount
        data[data == np.inf] = bincount

        alphas = alphavals.to_list()
        commonratios = commonratios.to_list()

        return alphas, commonratios, data

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
                 'alpha': (2*(data.quantile(0.75) - data.quantile(0.25)))/(data.shape[0]**(1/3))} 
                    # if alpha (from Freedman Diaconis) equals zero, then bin width is zero.
    
    nfeats = bin_stats['size'][1] 
    datalen = bin_stats['size'][0]
   
    # COLUMNS OF DATA FALL UNDER THESE FOUR RANGE DESCRIPTIONS
    positiverange = np.where((bin_stats['min'] >= 0) &(bin_stats['max'] > 0))[0]
    negativerange = np.where((bin_stats['min'] < 0) & (bin_stats['max'] <= 0))[0]
    neg2posrange =  np.where((bin_stats['min'] < 0) & (bin_stats['max'] > 0))[0]
    zeroalpha = np.where(bin_stats['alpha'] == 0)[0]

    # VALUES IN ZEROALPHA ARE DROPPED, NEED TO UPDATE BIN_STATS
    lenzeroalpha = len(zeroalpha)
    nfeats = nfeats-lenzeroalpha
    bin_stats['size'] = (datalen, nfeats-lenzeroalpha)
    for val in zeroalpha:
        bin_stats['min'].drop(column_names[val])
        bin_stats['max'].drop(column_names[val])
        bin_stats['alpha'].drop(column_names[val])
    
    # FIND COLUMNS THAT OVERLAP WITH ZEROALPHA(bin width is zero)
    POSoverlap = np.intersect1d(zeroalpha, positiverange, assume_unique = True, return_indices=True)
    NEGoverlap = np.intersect1d(zeroalpha, negativerange, assume_unique = True, return_indices=True)
    NEG2POSoverlap = np.intersect1d(zeroalpha, neg2posrange, assume_unique=True, return_indices=True)
    
    # REMOVE COLUMNS THAT OVERLAP WITH ZEROALPHA
    positiverange = np.delete(positiverange, POSoverlap[2])
    negativerange = np.delete(negativerange, NEGoverlap[2])
    NEG2POSoverlap = np.delete(neg2posrange, NEG2POSoverlap[2])

    # CREATING NEW DATA FRAMES OF THE DIFFERENT RANGE DESCRIPTIONS
        # Columns of data with a bin width value of zero is dropped in new dataframe. 
    positivedf = data.iloc[:, positiverange]
    negativedf = data.iloc[:, negativerange]
    neg2posdf = data.iloc[:, neg2posrange]
    # zerodf = data.iloc[:, zeroalpha]

    # COLLECTING NAMES OF COLUMNS OF THE DIFFERENT DATA FRAMES
    positivenames = positivedf.columns
    negativenames = negativedf.columns
    neg2posnames = neg2posdf.columns
    # zeronames = zerodf.columns 
    
    # POSITIVE RANGE
    alphaspos, commonratiospos, positivedf = dfzero2pos(positivedf, 
                                                        bin_stats['alpha'][positivenames], 
                                                        bin_stats['max'][positivenames],
                                                        datalen)
    yaxis = yaxis * len(positivenames)
    positivedf.reset_index(drop = True, inplace = True)

    # NEGATIVE RANGE
    alphasneg, commonratiosneg, negativedf = dfneg2zero(negativedf, 
                                                        -1*bin_stats['alpha'][negativenames], 
                                                        bin_stats['min'][negativenames],
                                                        datalen)
    yaxis = yaxis + ([bincount] * len(negativenames))
    negativedf.reset_index(drop = True, inplace = True)
    
    # NEGATIVE TO POSITIVE RANGE
    yvalues, alphasneg2pos, commonratiosneg2pos, neg2posdf = dfneg2pos(neg2posdf, 
                                                                       bin_stats['alpha'][neg2posnames], 
                                                                       bin_stats['min'][neg2posnames], 
                                                                       bin_stats['max'][neg2posnames],
                                                                       datalen)
    yaxis = yaxis + yvalues
    neg2posdf.reset_index(drop = True, inplace = True)

    # Concatenating alpha values and column bin sizes for the three dataframe transforms. 
    alphavals = alphaspos + alphasneg + alphasneg2pos
    column_bin_sizes = commonratiospos + commonratiosneg + commonratiosneg2pos
    

    # NEW DATA FRAME DROPS COLUMNS THAT HAS A BIN WIDTH VALUE OF ZERO
    data = pandas.concat([positivedf, negativedf, neg2posdf], axis=1)

    bins, bin_stats, log_index, log_dict, yaxis = bin_data(data, yaxis, "log", column_bin_sizes, bin_stats)

    return yaxis, bins, bin_stats, log_index, log_dict, column_bin_sizes, alphavals

def transform_data_linear(data,column_names):
    """ Bin the data
    
    Data from a pandas Dataframe is binned in two dimensions. Binning is performed by
    binning data in one column along one axis and another column is binned along the
    other axis. All combinations of columns are binned without repeats or transposition.
    There are only bincount number of bins in each dimension, and each bin is 1/bincount the size of the
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
    bin_stats = {'size': data.shape,
                 'min': data.min(),
                 'max': data.max()}
    column_bin_size = (bin_stats['max'] * (1 + 10**-6) - bin_stats['min'])/bincount

    # Transform data into bin positions for fast binning
    data = ((data - bin_stats['min'])/column_bin_size).apply(np.floor)

    bins, bin_stats, linear_index, linear_dict, yaxis = bin_data(data, yaxis, "linear", column_bin_size, bin_stats)
    
    return yaxis, bins, bin_stats, linear_index, linear_dict, column_bin_size, alphavals

""" 2. Plot Generation """
def format_ticks(out):
    """ Generate tick labels
    Polus Plots uses D3 to generate the plots. This function tries to mimic
    the formatting of tick labels. In place of using scientific notation a 
    scale prefix is appended to the end of the number. See _prefix comments 
    to see the suffixes that are used. Numbers that are larger or smaller 
    than 10**24 or 10**-24 respectively are not handled and may throw an 
    error. Values outside of this range do not currently have an agreed 
    upon prefix in the measurement science community.
    Inputs:
        out - the values of the ticks used in graph
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

    fticks = []
    convertprefix = []
    for i in out:
        formtick = "%#.3f" % i
        decformtick = '%.2e' % Decimal(formtick)
        convertexponent = float(decformtick[-3:])
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
                newprefix = _prefix[int(convertexponent)]
                if i < 0:
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
                if i < 0:
                    formtick = str(decformtick[:5]) + newprefix
                else: 
                    formtick = str(decformtick[:4]) + newprefix
        else:
            if i < 0:
                formtick = str(decformtick[:5]) + _prefix[int(convertexponent)]
            else: 
                formtick = str(decformtick[:4]) + _prefix[int(convertexponent)]
        convertprefix.append(int(convertexponent))
        fticks.append(formtick)

    return fticks

# Create a custom colormap to mimick Polus Plots
def get_cmap():
    
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
             indexdict,
             alphavals,
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
        binsizes -- the bin width
        column_names - list of column names
        bin_stats - a list containing the min,max values of each column
        fig - pregenerated figure
        ax - pregenerated axis
        data - pregenerated heatmap bbox artist
        axiszero -- where the x = 0 and y =0 in the graph
        typegraph -- sepcifies whether it is linear or log
    Outputs:
        hmap - A numpy array containing pixels of the heatmap
    """
    if col2>col1:
        d = np.squeeze(bins[indexdict[col1, col2],:,:])
        r = col1
        c = col2
    elif col2<col1:
        d = np.transpose(np.squeeze(bins[indexdict[(col1, col2)],:,:]))
        r = col2
        c = col1
    else:
        d = np.zeros((bincount,bincount))
        r = col1
        c = col2

    data.set_data(np.ceil(d/d.max() *255))
    data.set_clim(0, 255)

    sizefont = 12 
    axlabel = fig.axes[1]
    aylabel = fig.axes[2]
    
    # This is to decrease the size of the title labels if the name is too large (X AXIS LABEL)
    if len(axlabel.texts) == 0:
        axlabel.text(0.5, 0.5, "\n".join(wrap(column_names[c], 60)), va = 'center', ha = 'center', fontsize = sizefont, wrap = True)
        bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = sizefont - 1
        while (bbxtext.x0 < 0 or bbxtext.x1 > CHUNK_SIZE) or (bbxtext.y0 < 0 or bbxtext.y1 > CHUNK_SIZE*.075):
            axlabel.texts[0].set_fontsize(decreasefont)
            bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
            decreasefont = decreasefont - 1 
    else:
        axlabel.texts[0].set_text("\n".join(wrap(column_names[c], 60)))
        axlabel.texts[0].set_fontsize(sizefont)
        bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = sizefont - 1
        while (bbxtext.x0 < 0 or bbxtext.x1 > CHUNK_SIZE) or (bbxtext.y0 < 0 or bbxtext.y1 > (CHUNK_SIZE*.075)):
            axlabel.texts[0].set_fontsize(decreasefont)
            bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
            decreasefont = decreasefont - 1 

    # This is to decrease the size of the title labels if the name is too large (Y AXIS LABEL)
    if len(aylabel.texts) == 0:
        aylabel.text(0.5, 0.5, "\n".join(wrap(column_names[r], 60)), va = 'center', ha = 'center', fontsize = sizefont, rotation = 90, wrap = True)
        bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = sizefont - 1
        while (bbytext.y0 < 0 or bbytext.y1 > CHUNK_SIZE) or (bbytext.x0 < 0 or bbytext.x1 > (CHUNK_SIZE*.075)):
            aylabel.texts[0].set_fontsize(decreasefont)
            bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
            decreasefont = decreasefont - 1 

    else:
        aylabel.texts[0].set_text("\n".join(wrap(column_names[r], 60)))
        aylabel.texts[0].set_fontsize(sizefont)
        bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = sizefont - 1
        while (bbytext.y0 < 0 or bbytext.y1 > CHUNK_SIZE) or (bbytext.x0 < 0 or bbytext.x1 > (CHUNK_SIZE*.075)):
            aylabel.texts[0].set_fontsize(decreasefont)
            bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
            decreasefont = decreasefont - 1 
    
    # Ticks are formatted differently for both log and linearly scaled graphs. 
    if typegraph == "linear":
        # Calculating the value of each tick in the graph (fixed width)
        fmin = bin_stats['min'][column_names[c]]
        fmax = bin_stats['max'][column_names[c]]
        tick_vals = [t for t in np.arange(fmin, fmax,(fmax-fmin)/(numticks-1))]
        tick_vals.append(fmax)
        ax.set_xticklabels(format_ticks(tick_vals), rotation=45, fontsize = 5, ha='right')

        # Calculating the value of each tick in the graph (fixed width)
        fmin = bin_stats['min'][column_names[r]]
        fmax = bin_stats['max'][column_names[r]]
        tick_vals = [t for t in np.arange(fmin, fmax,(fmax-fmin)/(numticks-1))]
        tick_vals.append(fmax)
        ax.set_yticklabels(format_ticks(tick_vals), fontsize=5, ha='right')

    if typegraph == "log":
        # Calculating the value of each tick in the graph 
        yaxis = axiszero[c]
        commonratio = binsizes[c]
        alphavalue = alphavals[c]
        tick_vals = [(alphavalue*(commonratio**(t-yaxis))) if yaxis<t else (-1*(alphavalue*(commonratio**(yaxis-t))) if yaxis>t else 0) 
            for t in np.arange(0,bincount,bincount/(numticks-1))]
        if yaxis < bincount:
            tick_vals.append(alphavalue*(commonratio**(bincount-yaxis)))
        elif yaxis > bincount:
            tick_vals.append(-1*(alphavalue*(commonratio**(yaxis-bincount))))
        else:
            tick_vals.append(0)
        ax.set_xticklabels(format_ticks(tick_vals),rotation=45, fontsize = 5, ha='right')

        # Calculating the value of each tick in the graph 
        yaxis = axiszero[r]
        commonratio = binsizes[r]
        alphavalue = alphavals[r]
        tick_vals = [(alphavalue*(commonratio**(t-yaxis))) if yaxis<t else (-1*(alphavalue*(commonratio**(yaxis-t))) if yaxis>t else 0) 
            for t in np.arange(0,bincount,bincount/(numticks-1))]
        if yaxis < bincount:
            tick_vals.append(alphavalue*(commonratio**(bincount-yaxis)))
        elif yaxis > bincount:
            tick_vals.append(-1*(alphavalue*(commonratio**(yaxis-bincount))))
        else:
            tick_vals.append(0)
        ax.set_yticklabels(format_ticks(tick_vals), fontsize = 5, ha='right')

    # drawing the x axis and y axis lines on the graphs
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
    datacolor = ax.pcolorfast(np.zeros((bincount, bincount),np.uint64),cmap=cmap)
    ticks = [(t+0.5) for t in range(0, bincount+1, int(bincount/(numticks - 1)))]

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
def _get_higher_res(typegraph, S,info,cnames, outpath,out_file,indexscale,indexdict,binstats, binsizes, axiszero, alphavals, X=None,Y=None):
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
                             indexdict,
                             alphavals,
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
                                                   indexdict,
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
                                               indexdict,
                                               binstats,
                                               binsizes,
                                               axiszero,
                                               alphavals,
                                               X=subgrid_dims[0][x:x+2],
                                               Y=subgrid_dims[1][y:y+2])
                image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],:] = _avg2(sub_image)
                del sub_image

    # Write the chunk
    outpath = Path(outpath).joinpath('{}_files'.format(out_file),str(S))
    outpath.mkdir(exist_ok=True)
    imageio.imwrite(outpath.joinpath('{}_{}.png'.format(int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE))),image,format='PNG-FI',compression=1)
    logger.debug('Finished building tile (scale,X,Y): ({},{},{})'.format(S,int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE)))
    return image

# This function performs the same operation as _get_highe_res, except it uses multiprocessing to grab higher
# resolution layers at a specific layer.
def _get_higher_res_par(typegraph, S,info, cnames, outpath,out_file,indexscale, indexdict, binstats, binsizes, axiszero, alphavals, X=None,Y=None):
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
                             indexdict,
                             alphavals,
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
                                                                           indexdict,
                                                                           binstats,
                                                                           binsizes,
                                                                           axiszero,
                                                                           alphavals,
                                                                           subgrid_dims[0][x:x+2],
                                                                           subgrid_dims[1][y:y+2])))
            for y in range(0,len(subgrid_dims[1])-1):
                y_ind = [int(subgrid_dims[1][y] - subgrid_dims[1][0]),int(subgrid_dims[1][y+1] - subgrid_dims[1][0])]
                y_ind = [int(np.ceil(yi/2).astype('int')) for yi in y_ind]
                for x in range(0,len(subgrid_dims[0])-1):
                    x_ind = [int(subgrid_dims[0][x] - subgrid_dims[0][0]),int(subgrid_dims[0][x+1] - subgrid_dims[0][0])]
                    x_ind = [int(np.ceil(xi/2).astype('int')) for xi in x_ind]
                    image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],:] = _avg2((subgrid_images[int(y*(len(subgrid_dims[0])-1) + x)]).get())
        del subgrid_images

    # Write the chunk
    outpath = Path(outpath).joinpath('{}_files'.format(out_file),str(S))
    outpath.mkdir(exist_ok=True)
    imageio.imwrite(outpath.joinpath('{}_{}.png'.format(int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE))),image,format='PNG-FI',compression=1)
    logger.debug('Finished building tile (scale,X,Y): ({},{},{})'.format(S,int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE)))
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

    parser.add_argument('--outDir',
                        dest='outDir',
                        type=str,
                        help='Path to output images.',
                        required=True
                        )
    
    parser.add_argument('--bincount',
                        dest='bin_count',
                        type=int,
                        help='Number of bins',
                        required=True
                        )

    """ Get the input arguments """
    args = parser.parse_args()

    input_path = args.inpDir
    output_path = Path(args.outDir)
    bincount = args.bin_count

    logger.info('inpDir = {}'.format(input_path))
    logger.info('outDir = {}'.format(output_path))

    linear_logger = logging.getLogger("main.LINEAR")
    linear_logger.setLevel(logging.INFO)

    log_logger = logging.getLogger("main.LOG")
    log_logger.setLevel(logging.INFO)

    # Get the path to each csv file in the collection
    input_files = [str(f.absolute()) for f in Path(input_path).iterdir() if ''.join(f.suffixes)=='.csv']

    for f in input_files:

        global bins
        folder = Path(f)

        # Set the file path folder
        folder_linear = folder.name.replace('.csv','_linear')
        folder_log = folder.name.replace('.csv','_log')

        # Load the data
        linear_logger.info('Loading LINEAR csv: {}'.format(f))
        data_linear, cnames = load_csv(f)
        column_names = data_linear.columns
        linear_logger.info('Done loading LINEAR csv!')

        # Load the data
        log_logger.info('Loading LOG csv: {}'.format(f))
        data_log = data_linear
        log_logger.info('Done LOG loading csv!')

        # Processes for LINEAR SCALED GRAPHS
        linear_logger.info('Processing: {}'.format(folder_linear))

        # Bin the data
        linear_logger.info('Binning data for {} LINEAR features...'.format(column_names.size))
        # start_linear = time.time()
        yaxis_linear, bins, bin_stats, linear_index, linear_dict, linear_binsizes, alphavals_linear = transform_data_linear(data_linear,column_names)
        # end_linear = time.time()
        # totaltime_linear = end_linear - start_linear
        del data_linear # get rid of the original data to save memory

        # Generate the default figure components
        linear_logger.info('Generating colormap and default figure...')
        cmap_linear = get_cmap()
        fig, ax, datacolor = get_default_fig(cmap_linear)
        linear_logger.info('Done!')

        # Generate the dzi file
        linear_logger.info('Generating pyramid LINEAR metadata...')
        info_linear = metadata_to_graph_info(bins, output_path,folder_linear, linear_index)
        linear_logger.info('Done!')

        linear_logger.info('Writing LINEAR layout file...!')
        write_csv(cnames,linear_index,info_linear,output_path,folder_linear)
        linear_logger.info('Done!')

        # Create the pyramid
        linear_logger.info('Building LINEAR pyramids...')
        image_linear = _get_higher_res("linear", 0, info_linear,column_names, output_path,folder_linear,linear_index, linear_dict, bin_stats, linear_binsizes, yaxis_linear, alphavals_linear)

        
        del image_linear
        del info_linear
        del yaxis_linear
        del bin_stats
        del linear_index
        del linear_binsizes
        del alphavals_linear
        bins = 0

        # Processes for LOG SCALED GRAPHS
        log_logger.info('Processing: {}'.format(folder_log))
        
        # Bin the data
        log_logger.info('Binning data for {} LOG features...'.format(column_names.size))
        start_log = time.time()
        yaxis_log, bins, log_bin_stats, log_index, log_dict, log_binsizes, alphavals_log = transform_data_log(data_log, column_names)
        end_log = time.time()
        totaltime_log = end_log - start_log
        del data_log # get rid of the original data to save memory

        # Generate the default figure components
        log_logger.info('Generating colormap and default figure...')
        cmap_log = get_cmap()
        fig, ax, datacolor = get_default_fig(cmap_log)
        log_logger.info('Done!')

        # Generate the dzi file
        log_logger.info('Generating pyramid LOG metadata...')
        info_log = metadata_to_graph_info(bins, output_path,folder_log, log_index)
        log_logger.info('Done!')

        log_logger.info('Writing LOG layout file...!')
        write_csv(cnames, log_index, info_log, output_path, folder_log)
        log_logger.info('Done!')

        # Create the pyramid
        log_logger.info('Building LOG pyramid...')
        image_log = _get_higher_res("log", 0, info_log, column_names, output_path, folder_log, log_index, log_dict, log_bin_stats, log_binsizes, yaxis_log, alphavals_log)

