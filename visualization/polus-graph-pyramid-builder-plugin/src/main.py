import pandas, multiprocessing, argparse, logging, matplotlib, copy, imageio
from pathlib import Path
from multiprocessing import Pool

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
matplotlib.use('agg')

import math
import numpy as np
import decimal
from decimal import Decimal

from textwrap import wrap

# Chunk Scale
CHUNK_SIZE = 1024

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

def bin_data(data, bin_stats):
    """ This function bins the data 
    Inputs:
        data - pandas dataframe of data
        bin_stats - stats of the data 
    Outputs:
        bins - binned data ranging from (0, bincount)
        graph_index - Numeric value of column index from original csv
        graph_dict - a dictionary containing the indexes of graphs
    """

    column_names = data.columns
    nfeats = data.shape[1] 
    nrows = data.shape[0]

    # Handle NaN values
    data_ind = pandas.notnull(data)
    data[~data_ind] = 255
    
    data = data.astype(np.uint16)       # cast to save memory
    data[data>=bincount] = bincount - 1 # in case of numerical precision issues

    
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
            ind2 = np.nonzero(np.diff(SortedFeats))[0]                              # nonzeros are cumulative sum of all bin values
            ind2 = np.append(ind2,SortedFeats.size-1)
            rows = (SortedFeats[ind2]/bincount).astype(np.uint8)    # calculate row from linear index
            cols = np.mod(SortedFeats[ind2],bincount)               # calculate column from linear index
            counts = np.diff(ind2)                                  # calculate the number of values in each bin
            
            bins[i,rows[0],cols[0]] = ind2[0] + 1 
            bins[i,rows[1:],cols[1:]] = counts 
            graph_index.append([feat1,feat2])
            i = i + 1

    return bins, graph_index, graph_dict

def transform_data(data,column_names, typegraph):
    """ Bin the data
    
    Data from a pandas Dataframe is binned in two dimensions. Binning is performed by
    binning data in one column along one axis and another column is binned along the
    other axis. All combinations of columns are binned without repeats or transposition.
    There are only bincount number of bins in each dimension, and each bin is 1/bincount the size of the
    difference between the maximum and minimum of each column. 
    If the data needs to be logarithmically scaled, then the data is transformed by the algorithm presented
    in this paper: https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001
    Inputs:
        data - A pandas Dataframe, with nfeats number of columns
        column_names - Names of Dataframe columns
        typegraph - Defines whether logarithmic scale or linear scalef
    Outputs:
        bins - A numpy matrix that has shape (int((nfeats**2 - nfeats)/2),bincount,bincount)
        bin_feats - A list containing the minimum and maximum values of each column
        index - Numeric value of column index from original csv
        diction - a dictionary containing the indexes of graphs
    """

    nfeats = len(column_names)

    # If logarithmic, need to transform the data
    # https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001
    # Adjusts for behavior near zero

    if typegraph == "log":
        C = 1/np.log(10)# Derivative of Natural Log e, d(ln(x))/dx = 1/x
        data = data.astype(np.float64)
        data = np.sign(data) * np.log10(1 + (abs(data/C)))

    bin_stats = {'min': data.min(),
                 'max': data.max(),
                 'binwidth': (data.max()-data.min()+10**-6)/bincount}    
 
    
    # Transform data into bin positions for fast binning
    data = ((data - bin_stats['min'])/bin_stats['binwidth']).apply(np.floor)

    bins, index, diction = bin_data(data, bin_stats)
    return bins, bin_stats, index, diction

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
             column_names,
             bin_stats,
             fig,
             ax,
             data,
             typegraph):
    """ Generate a heatmap
    Generate a heatmap of data for column 1 against column 2.
    Inputs:
        col1 - the column plotted on the y-axis
        col2 - column plotted on the x-axis
        indexdict - a dictionary containing the indexes of graphs
        column_names - list of column names
        bin_stats - a list containing the min,max values of each column
        fig - pregenerated figure
        ax - pregenerated axis
        data - p  regenerated heatmap bbox artist
        typegraph - specifies whether the data is log scaled or linearly scaled
    Outputs:
        hmap - A numpy array containing pixels of the heatmap
    """
    def keepdecreasing(labeltexts0, decreasefont, bbxtext):
        """ This function decreases the size of the labels if its too big """
        labeltexts0.set_fontsize(decreasefont)
        bbxtext = labeltexts0.get_window_extent(renderer = fig.canvas.renderer)
        decreasefont = decreasefont - 1 
        return bbxtext, decreasefont

    def calculateticks(ticks, bin_width, fmin, typegraph):
        """ This functio n calculates the tick values for the graphs """

        if typegraph == "linear":
            tick_vals = [t for t in ticks*bin_width+fmin]
        if typegraph == "log": 
            C = 1/np.log(10)
            tick_vals = [np.sign(t)*C*(-1+(10**abs(t))) for t in ticks*bin_width+fmin]
        return tick_vals

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

    data.set_data(np.ceil(d/d.max() * 255))
    data.set_clim(0, 255)

    sizefont = 12 
    axlabel = fig.axes[1]
    aylabel = fig.axes[2] 

    cname_c = column_names[c]
    cname_r = column_names[r]

    
    # This is to decrease the size of the title labels if the name is too large (X AXIS LABEL)
    if len(axlabel.texts) == 0:
        axlabel.text(0.5, 0.5, "\n".join(wrap(cname_c, 60)), va = 'center', ha = 'center', fontsize = sizefont, wrap = True)
    else:
        axlabeltext0 = axlabel.texts[0]
        axlabeltext0.set_text("\n".join(wrap(cname_c, 60)))
        axlabeltext0.set_fontsize(sizefont)

    bbxtext = (axlabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
    decreasefont = sizefont - 1
    while (bbxtext.x0 < 0 or bbxtext.x1 > CHUNK_SIZE) or (bbxtext.y0 < 0 or bbxtext.y1 > (CHUNK_SIZE*.075)):
        bbxtext, decreasefont = keepdecreasing(axlabel.texts[0], decreasefont, bbxtext)

    # This is to decrease the size of the title labels if the name is too large (Y AXIS LABEL)
    if len(aylabel.texts) == 0:
        aylabel.text(0.5, 0.5, "\n".join(wrap(cname_r, 60)), va = 'center', ha = 'center', fontsize = sizefont, rotation = 90, wrap = True)
    else:
        aylabeltext0 = aylabel.texts[0]
        aylabeltext0.set_text("\n".join(wrap(cname_r, 60)))
        aylabeltext0.set_fontsize(sizefont)

    bbytext = (aylabel.texts[0]).get_window_extent(renderer = fig.canvas.renderer)
    decreasefont = sizefont - 1
    while (bbytext.y0 < 0 or bbytext.y1 > CHUNK_SIZE) or (bbytext.x0 < 0 or bbytext.x1 > (CHUNK_SIZE*.075)):
        bbytext, decreasefont = keepdecreasing(aylabel.texts[0], decreasefont, bbytext)
    
    while len(ax.lines) > 0:
        ax.lines[-1].remove()

    # Calculating the value of each tick in the graph (fixed width)
    fmin_c = bin_stats['min'][cname_c]
    fmax_c = bin_stats['max'][cname_c]
    binwidth_c = bin_stats['binwidth'][cname_c]
    tick_vals_c= calculateticks(ax.get_xticks(), binwidth_c, fmin_c, typegraph)
    if fmin_c < 0: # draw x=0
        ax.axvline(x=abs(fmin_c)/binwidth_c)
    ax.set_xticklabels(format_ticks(tick_vals_c), rotation=45, fontsize = 5, ha='right')

    # Calculating the value of each tick in the graph (fixed width)
    fmin_r = bin_stats['min'][cname_r]
    fmax_r = bin_stats['max'][cname_r]
    binwidth_r = bin_stats['binwidth'][cname_r]
    tick_vals_r = calculateticks(ax.get_yticks(), binwidth_r, fmin_r, typegraph)
    if fmin_r < 0: # draw y=0
        ax.axhline(y=abs(fmin_r)/binwidth_r)
    ax.set_yticklabels(format_ticks(tick_vals_r), fontsize=5, ha='right')

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
    ticks = [t for t in range(0, bincount+1, int(bincount/(10)))]

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

    return fig, ax, datacolor

""" 3. Pyramid generation functions """

def _avg2(image):
    """ Average pixels with optical field of 2x2 and stride 2 """
    
    # Convert 32-bit pixels to prevent overflow during averaging
    image = image.astype(np.uint32)
    imageshape0 = image.shape[0]
    imageshape1 = image.shape[1]
    # Get the height and width of each image to the nearest even number
    y_max = imageshape0 - imageshape0 % 2
    x_max = imageshape1 - imageshape1 % 2
    
    # Perform averaging
    avg_img = np.zeros(np.ceil([image.shape[0]/2,image.shape[1]/2,image.shape[2]]).astype(np.uint32))
    for z in range(4):
        avg_img[0:int(y_max/2),0:int(x_max/2),z]= (image[0:y_max-1:2,0:x_max-1:2,z] + \
                                                   image[1:y_max:2,0:x_max-1:2,z] + \
                                                   image[0:y_max-1:2,1:x_max:2,z] + \
                                                   image[1:y_max:2,1:x_max:2,z]) / 4
        
    # The next if statements handle edge cases if the height or width of the
    # image has an odd number of pixels
    if y_max != imageshape0:
        for z in range(3):
            avg_img[-1,:int(x_max/2),z] = (image[-1,0:x_max-1:2,z] + \
                                           image[-1,1:x_max:2,z]) / 2
    if x_max != imageshape1:
        for z in range(4):
            avg_img[:int(y_max/2),-1,z] = (image[0:y_max-1:2,-1,z] + \
                                           image[1:y_max:2,-1,z]) / 2
    if y_max != imageshape0 and x_max != imageshape1:
        for z in range(4):
            avg_img[-1,-1,z] = image[-1,-1,z]
    return avg_img

def metadata_to_graph_info(outPath,outFile, ngraphs):
    
    # Create an output path object for the info file
    op = Path(outPath).joinpath("{}.dzi".format(outFile))

    # create an output path for the images
    of = Path(outPath).joinpath('{}_files'.format(outFile))
    of.mkdir(exist_ok=True)
    
    # Get metadata info from the bfio reader
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


def _get_higher_res(S,info,cnames, outpath,out_file,indexscale,indexdict,binstats, typegraph, X=None,Y=None):
    """
    The following function builds the image pyramid at scale S by building up only
    the necessary information at high resolution layers of the pyramid. So, if 0 is
    the original resolution of the image, getting a tile at scale 2 will generate
    only the necessary information at layers 0 and 1 to create the desired tile at
    layer 2. This function is recursive and can be parallelized.
    Inputs:
        S - current scale
        info - dictionary of scale information
        outpath - directory for all outputs
        out_file - directory for current dataset
        indexscale - index of the graph 
        binstats - stats for the binned data
        typegraph - specifies whether the data is linear or logarithmically scaled
    Outputs:
        DeepZoom format of images.
    """

    # Get the scale info
    num_scales = len(info['scales'])
    scale_info = info['scales'][num_scales-S-1]

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
    if S==num_scales-1:
        index = int((int(Y[0]/CHUNK_SIZE) + int(X[0]/CHUNK_SIZE) * info['rows']))
        if index>=len(indexscale):
            image = np.ones((CHUNK_SIZE,CHUNK_SIZE,4),dtype=np.uint8) * (bincount + 55)
        else:
            image = gen_plot(col1=indexscale[index][0],
                                col2=indexscale[index][1],
                                indexdict=indexdict,
                                column_names=cnames,
                                bin_stats=binstats,
                                fig=fig,
                                ax=ax,
                                data=datacolor,
                                typegraph=typegraph)
                            
    else:
        # Set the subgrid dimensions
        subgrid_dimX = list(np.arange(2*X[0], 2*X[1], CHUNK_SIZE).astype('int'))
        subgrid_dimX.append(2*X[1])
        subgrid_dimY = list(np.arange(2*Y[0], 2*Y[1], CHUNK_SIZE).astype('int'))
        subgrid_dimY.append(2*Y[1])
        
        
        for y in range(0,len(subgrid_dimY)-1):
            subgrid_Y_ind0 = np.ceil((subgrid_dimY[y] - subgrid_dimY[0])/2).astype('int')
            subgrid_Y_ind1 = np.ceil((subgrid_dimY[y+1] - subgrid_dimY[0])/2).astype('int')
            for x in range(0,len(subgrid_dimX)-1):
                subgrid_X_ind0 = np.ceil((subgrid_dimX[x] - subgrid_dimX[0])/2).astype('int')
                subgrid_X_ind1 = np.ceil((subgrid_dimX[x+1] - subgrid_dimX[0])/2).astype('int')
                if S==(num_scales - 6): #to use multiple processors to compute faster.
                    sub_image = _get_higher_res_par(S=S+1,
                                                    info=info,
                                                    cnames=cnames,
                                                    outpath=outpath,
                                                    out_file=out_file,
                                                    indexscale=indexscale,
                                                    indexdict=indexdict,
                                                    binstats=binstats,
                                                    typegraph=typegraph, 
                                                    X=subgrid_dimX[x:x+2],
                                                    Y=subgrid_dimY[y:y+2])
                else:
                    sub_image = _get_higher_res(S=S+1,
                                                info=info,
                                                cnames=cnames,
                                                outpath=outpath,
                                                out_file=out_file,
                                                indexscale=indexscale,
                                                indexdict=indexdict,
                                                binstats=binstats,
                                                typegraph=typegraph, 
                                                X=subgrid_dimX[x:x+2],
                                                Y=subgrid_dimY[y:y+2])
                                                
                image[subgrid_Y_ind0:subgrid_Y_ind1, subgrid_X_ind0:subgrid_X_ind1,:] = _avg2(sub_image)
                del sub_image

    # Write the chunk
    outpath = Path(outpath).joinpath('{}_files'.format(out_file),str(S))
    outpath.mkdir(exist_ok=True)
    imageio.imwrite(outpath.joinpath('{}_{}.png'.format(int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE))),image,format='PNG-FI',compression=1)
    logger.info('Finished building tile (scale,X,Y): ({},{},{})'.format(S,int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE)))
    return image

# This function performs the same operation as _get_highe_res, except it uses multiprocessing to grab higher
# resolution layers at a specific layer.
def _get_higher_res_par(S,info, cnames, outpath,out_file,indexscale, indexdict, binstats, typegraph, X=None,Y=None):
    # Get the scale info
    num_scales = len(info['scales'])
    scale_info = info['scales'][num_scales-S-1]

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
            image = gen_plot(col1=indexscale[index][0],
                             col2=indexscale[index][1],
                             indexdict=indexdict,
                             column_names=cnames,
                             bin_stats=binstats,
                             fig=fig,
                             ax=ax,
                             data=datacolor,
                             typegraph=typegraph)

    else:
        # Set the subgrid dimensions
        subgrid_dimX = list(np.arange(2*X[0], 2*X[1], CHUNK_SIZE).astype('int'))
        subgrid_dimX.append(2*X[1])
        subgrid_dimY = list(np.arange(2*Y[0], 2*Y[1], CHUNK_SIZE).astype('int'))
        subgrid_dimY.append(2*Y[1])

        subgrid_images = []
        
        with Pool(processes=np.min(4,initial=multiprocessing.cpu_count())) as pool:
            for y in range(0,len(subgrid_dimY)-1):
                subgrid_Y_ind0 = np.ceil((subgrid_dimY[y] - subgrid_dimY[0])/2).astype('int')
                subgrid_Y_ind1 = np.ceil((subgrid_dimY[y+1] - subgrid_dimY[0])/2).astype('int')
                for x in range(0,len(subgrid_dimX)-1):
                    subgrid_X_ind0 = np.ceil((subgrid_dimX[x] - subgrid_dimX[0])/2).astype('int')
                    subgrid_X_ind1 = np.ceil((subgrid_dimX[x+1] - subgrid_dimX[0])/2).astype('int')
                    subgrid_images.append(pool.apply_async(_get_higher_res,(S+1,
                                                                           info,
                                                                           cnames,
                                                                           outpath,
                                                                           out_file,
                                                                           indexscale,
                                                                           indexdict,
                                                                           binstats,
                                                                           typegraph,
                                                                           subgrid_dimX[x:x+2],
                                                                           subgrid_dimY[y:y+2])))
                    image[subgrid_Y_ind0:subgrid_Y_ind1,subgrid_X_ind0:subgrid_X_ind1,:] = _avg2((subgrid_images[y*(len(subgrid_dimX)-1) + x]).get())
        
        del subgrid_images

    # Write the chunk
    outpath = Path(outpath).joinpath('{}_files'.format(out_file),str(S))
    outpath.mkdir(exist_ok=True)
    imageio.imwrite(outpath.joinpath('{}_{}.png'.format(int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE))),image,format='PNG-FI',compression=1)
    logger.info('Finished building tile (scale,X,Y): ({},{},{})'.format(S,int(X[0]/CHUNK_SIZE),int(Y[0]/CHUNK_SIZE)))
    return image

def write_csv(cnames,index,f_info,out_path,out_file):
    """ This function writes the csv file necessary for the Deep Zoom format """

    header = 'dataset_id, x_axis_id, y_axis_id, x_axis_name, y_axis_name, title, length, width, global_row, global_col\n'
    line = '{:d}, {:d}, {:d}, {:s}, {:s}, default title, {:d}, {:d}, {:d}, {:d}\n'
    l_ind = 0
    with open(str(Path(out_path).joinpath(out_file+'.csv').absolute()),'w') as writer:
        writer.write(header)
        for ind in index:
            ind1 = ind[1]
            ind0 = ind[0]
            writer.write(line.format(1,
                                     cnames[ind1][1],
                                     cnames[ind0][1],
                                     cnames[ind1][0],
                                     cnames[ind0][0],
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
    cmap = get_cmap()
    fig, ax, datacolor = get_default_fig(cmap)
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
            bins, bin_stats, data_index, data_dict = transform_data(data,column_names, scale)

            # Generate the dzi file
            loggers[scale].info('Generating pyramid {} metadata...'.format(scale.upper()))
            ngraphs = len(data_index)
            info_data = metadata_to_graph_info(output_path,folder_name, ngraphs)
            loggers[scale].info('Done!')

            loggers[scale].info('Writing {} layout file...!'.format(scale.upper()))
            write_csv(cnames,data_index,info_data,output_path,folder_name)
            loggers[scale].info('Done!')

            # Create the pyramid
            loggers[scale].info('Building {} pyramids...'.format(scale.upper()))
            image_data = _get_higher_res(0, info_data,column_names, output_path,folder_name,data_index, data_dict, bin_stats, scale)
            loggers[scale].info('Done!')
