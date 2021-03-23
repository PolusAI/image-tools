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

def bin_data(data, yaxis, column_bin_size, bin_stats):
    """ This function bins the data """

    column_names = data.columns
    nfeats = bin_stats['size'][1] 
    datalen = bin_stats['size'][0]

    # Handle NaN values
    data_ind = pandas.notnull(data)
    data[~data_ind] = 255
    
    data = data.astype(np.uint16)       # cast to save memory
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
        
        if bin_stats['min'][feat1] >= 0:
            yaxis[feat1] = 0
        else:
            yaxis[feat1] = abs(bin_stats['min'][feat1])/column_bin_size[feat1] + 1

        for feat2 in range(feat1 + 1, nfeats):
            graph_dict[(feat1, feat2)] = i
            name2 = column_names[feat2]
            
            feat2_tf = data[name2]
            feat2_tf = feat2_tf[data_ind[name1] & data_ind[name2]]
                      
            if feat2_tf.size<=1:
                continue
            
            # sort linear matrix indices
            SortedFeats = np.sort(feat1_tf[data_ind[name1] & data_ind[name2]] + feat2_tf)
            
            # Do math to get the indices
            ind2 = np.diff(SortedFeats)                       
            ind2 = np.nonzero(ind2)[0]                              # nonzeros are cumulative sum of all bin values
            ind2 = np.append(ind2,SortedFeats.size-1)
            rows = (SortedFeats[ind2]/bincount).astype(np.uint8)    # calculate row from linear index
            cols = np.mod(SortedFeats[ind2],bincount)               # calculate column from linear index
            counts = np.diff(ind2)                                  # calculate the number of values in each bin
            
            bins[i,rows[0],cols[0]] = ind2[0] + 1 
            bins[i,rows[1:],cols[1:]] = counts 
            graph_index.append([feat1,feat2])
            i = i + 1

    return bins, bin_stats, graph_index, graph_dict, yaxis

def transform_data(data,column_names, typegraph):
    """ Bin the data
    
    Data from a pandas Dataframe is binned in two dimensions. Binning is performed by
    binning data in one column along one axis and another column is binned along the
    other axis. All combinations of columns are binned without repeats or transposition.
    There are only bincount number of bins in each dimension, and each bin is 1/bincount the size of the
    difference between the maximum and minimum of each column.
    Inputs:
        data - A pandas Dataframe, with nfeats number of columns
        column_names - Names of Dataframe columns
        typegraph - Defines whether logarithmic scale or linear scale
    Outputs:
        bins - A numpy matrix that has shape (nfeats,nfeats,200,200)
        bin_feats - A list containing the minimum and maximum values of each column
        index - Numeric value of column index from original csv
    """

    nfeats = len(column_names)
    yaxis = np.zeros(nfeats, dtype=int)

    # If logarithmic, need to transform the data
    # https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001
    if typegraph == "log":
        C = 1/np.log(10)
        data = np.sign(data) * np.log10(1 + (abs(data/C)))
    bin_stats = {'size': data.shape,
                 'min': data.min(),
                 'max': data.max()}
    column_bin_size = (bin_stats['max'] * (1 + 10**-6) - bin_stats['min'])/bincount


    # Transform data into bin positions for fast binning
    data = ((data - bin_stats['min'])/column_bin_size).apply(np.floor)

    bins, bin_stats, index, diction, yaxis = bin_data(data, yaxis, column_bin_size, bin_stats)
    return yaxis, bins, bin_stats, index, diction, column_bin_size

""" 2. Plot Generation """
def format_ticks(out):
    """ Generate tick labels
    Polus Plots uses D3 to generate the plots. This function tries to mimic the
    formatting of tick labels. In place of using scientific notation a scale
    prefix is appended to the end of the number. See _prefix comments to see the
    suffixes that are used. Numbers that are larger or smaller than 10**24 or
    10**-24 respectively are not handled and may throw an error. Values outside
    of this range do not currently have an agreed upon prefix in the measurement
    science community.
        
    Inputs:
        out - the values of the ticks used in graph
    Outputs:
        fticks - a list of strings containing formatted tick labels
    """
    _prefix = {
        -24: 'y',  # yocto
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
             binsizes,
             column_names,
             bin_stats,
             fig,
             ax,
             data,
             axiszero):
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
        
    # The next if statements handle edge cases if the height or width of the
    # image has an odd number of pixels
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

"""
The following function builds the image pyramid at scale S by building up only
the necessary information at high resolution layers of the pyramid. So, if 0 is
the original resolution of the image, getting a tile at scale 2 will generate
only the necessary information at layers 0 and 1 to create the desired tile at
layer 2. This function is recursive and can be parallelized.
"""
def _get_higher_res(S,info,cnames, outpath,out_file,indexscale,indexdict,binstats, binsizes, axiszero, X=None,Y=None):
    # Get the scale info
    scale_info = None
    for res in info['scales']: 
        if int(res['key'])==S:
            scale_info = res
            break
    if scale_info==None:
        raise ValueError("No scale information for resolution {}.".format(S))
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
    image = np.zeros((int(Y[1]-Y[0]),int(X[1]-X[0]),4),dtype=np.uint8)
    
    # If requesting from the lowest scale, then just generate the graph
    if S==int(info['scales'][0]['key']):
        index = int((int(Y[0]/CHUNK_SIZE) + int(X[0]/CHUNK_SIZE) * info['rows']))
        if index>=len(indexscale):
            image = np.ones((CHUNK_SIZE,CHUNK_SIZE,4),dtype=np.uint8) * (bincount + 55)
        else:
            image = gen_plot(indexscale[index][0],
                             indexscale[index][1],
                             indexdict,
                             binsizes,
                             cnames,
                             binstats,
                             fig,
                             ax,
                             datacolor,
                             axiszero)
    else:
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]]]
        
        for dim in subgrid_dims:
            while dim[1]-dim[0] > CHUNK_SIZE:
                dim.insert(1,dim[0] + ((dim[1] - dim[0]-1)//CHUNK_SIZE) * CHUNK_SIZE)

        for y in range(0,len(subgrid_dims[1])-1):
            y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
            y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
            for x in range(0,len(subgrid_dims[0])-1):
                x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                if S==(info['scales'][0]['key'] - 5): #to use multiple processors to compute faster.
                    sub_image = _get_higher_res_par(S+1,
                                                    info,
                                                    cnames,
                                                    outpath,
                                                    out_file,
                                                    indexscale,
                                                    indexdict,
                                                    binstats,
                                                    binsizes,
                                                    axiszero,
                                                    X=subgrid_dims[0][x:x+2],
                                                    Y=subgrid_dims[1][y:y+2])
                else:
                    sub_image = _get_higher_res(S+1,
                                                info,
                                                cnames,
                                                outpath,
                                                out_file,
                                                indexscale,
                                                indexdict,
                                                binstats,
                                                binsizes,
                                                axiszero,
                                                X=subgrid_dims[0][x:x+2],
                                                Y=subgrid_dims[1][y:y+2])
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
def _get_higher_res_par(S,info, cnames, outpath,out_file,indexscale, indexdict, binstats, binsizes, axiszero, X=None,Y=None):
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
                             binsizes,
                             cnames,
                             binstats,
                             fig,
                             ax,
                             datacolor,
                             axiszero)
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
                                                                           cnames,
                                                                           outpath,
                                                                           out_file,
                                                                           indexscale,
                                                                           indexdict,
                                                                           binstats,
                                                                           binsizes,
                                                                           axiszero,
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
    logger.info('Finished building tile (scale,X,Y): ({},{},{})'.format(S,int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE)))
    return image

def write_csv(cnames,index,f_info,out_path,out_file):
    header = 'dataset_id, x_axis_id, y_axis_id, x_axis_name, y_axis_name, title, length, width, global_row, global_col\n'
    line = '{:d}, {:d}, {:d}, {:s}, {:s}, default title, {:d}, {:d}, {:d}, {:d}\n'
    l_ind = 0
    with open(str(Path(out_path).joinpath(out_file+'.csv').absolute()),'w') as writer:
        writer.write(header)
        for ind in index:
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
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Path to input images.', required=True)

    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Path to output images.', required=True)
    
    parser.add_argument('--bincount', dest='bin_count', type=int,
                        help='Number of bins', required=True)
    
    parser.add_argument('--scale', dest='scale', type=str,
                        help='Linear, Log, or Both', required=False)

    """ Get the input arguments """
    args = parser.parse_args()

    input_path = args.inpDir
    output_path = Path(args.outDir)
    bincount = args.bin_count
    scales = [args.scale.lower()]
    all_scales = ['linear','log']
    if scales[0] not in all_scales:
        scales = all_scales

    logger.info('inpDir = {}'.format(input_path))
    logger.info('outDir = {}'.format(output_path))

    # Set up the logger for each scale
    loggers = {}
    for scale in scales:
        loggers[scale] = logging.getLogger("main.{}".format(scale.upper()))
        loggers[scale].setLevel(logging.INFO)

    # Get the path to each csv file in the collection
    input_files = [str(f.absolute()) for f in Path(input_path).iterdir() if ''.join(f.suffixes)=='.csv']
    
    # Generate the default figure components
    logger.info('Generating colormap and default figure...')
    cmap_linear = get_cmap()
    fig, ax, datacolor = get_default_fig(cmap_linear)
    logger.info('Done!')

    for f in input_files:
        
        logger.info('Loading csv: {}'.format(f))
        data, cnames = load_csv(f)
        column_names = [c[0] for c in cnames]

        for scale in scales:
            
            # Set the file path folder
            folder_name = Path(f).name.replace('.csv','_{}'.format(scale))
            
            # Process for current scale
            loggers[scale].info('Processing: {}'.format(folder_name))

            # Bin the data
            loggers[scale].info('Binning data for {} {} features...'.format(len(column_names),scale.upper()))
            yaxis_data, bins, bin_stats, data_index, data_dict, data_binsizes = transform_data(data,column_names, scale)

            # Generate the dzi file
            loggers[scale].info('Generating pyramid {} metadata...'.format(scale.upper()))
            info_data = metadata_to_graph_info(bins, output_path,folder_name, data_index)
            loggers[scale].info('Done!')

            loggers[scale].info('Writing {} layout file...!'.format(scale.upper()))
            write_csv(cnames,data_index,info_data,output_path,folder_name)
            loggers[scale].info('Done!')

            # Create the pyramid
            loggers[scale].info('Building {} pyramids...'.format(scale.upper()))
            image_data = _get_higher_res(0, info_data,column_names, output_path,folder_name,data_index, data_dict, bin_stats, data_binsizes, yaxis_data)
            loggers[scale].info('Done!')
