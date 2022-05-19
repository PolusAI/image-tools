import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde

import imageio
imageio.plugins.freeimage.download()

import numpy as np
import pandas as pd

from textwrap import wrap

from decimal import Decimal

import logging
# Initialize the logger
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG', 'INFO'))
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("figures")
logger.setLevel(POLUS_LOG)

convert_tolog = lambda t : np.sign(t)*(1/np.log(10))*(-1+(10**abs(t)))

def format_ticks(out : list):
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

def get_cmap():
    """ This function gives the range of colors for the heatmaps generated. """

    cmap_values = [[1.0,1.0,1.0,1.0]]
    cmap_values.extend([[r/255,g/255,b/255,1] for r,g,b in zip(np.arange(0,255,2),
                                                        np.arange(153,255+1/128,102/126),
                                                        np.arange(34+1/128,0,-34/126))])
    cmap_values.extend([[r/255,g/255,b/255,1] for r,g,b in zip(np.arange(255,136-1/128,-119/127),
                                                        np.arange(255,0,-2),
                                                        np.arange(0,68+1/128,68/127))])
    cmap = ListedColormap(cmap_values)
    return cmap

def get_dtype(nrows : int):
    """ This functions helps to figure out the data type for the output image in heatmaps. 
        This can help save memory by not always assigning to np.uint64 
        
    Inputs:
        nrows - the max of datapoints that can be in a bin
    """

    if nrows < 2**8:
        dtype = np.uint8
    elif nrows < 2**16:
        dtype = np.uint16
    elif nrows < 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64

    return dtype


class Figures():

    def __init__(self, bins : int, 
                       CHUNK_SIZE : int, 
                       color : str, 
                       scale : str, 
                       stats : dict, 
                       output_dir : str):
        """ This function intializes the graphs to build the data. 
        Some of the parameters of the graphs are consistent amoung a number of graphs.
        
        Inputs: 
            bins       - this number specifies the number of bins for heatmaps and histograms
            CHUNK_SIZE - the size of one image 
            color      - color of graphs
            scale      - lienar or log, this affects the formatting
            stats      - a dictionary containing the stats of the data that is being plotted
            output_dir - location of where all the images are saved
        """

        self.bins = bins
        self.nticks = 11
        self.stats = stats

        self.CHUNK_SIZE = CHUNK_SIZE
        self.fig, self.ax = plt.subplots(dpi=int(self.CHUNK_SIZE/4),figsize=(4,4),tight_layout={'h_pad':1,'w_pad':1})

        self.cmap = get_cmap()

        self.color = color
        self.scale = scale

        self.ax.set_xlabel(" ")
        self.ax.set_ylabel(" ")

        self.convert_tolog = convert_tolog

        self.output_dir = output_dir

    def format_axis(self,
                    label_x          : str,          label_y          : str,
                    ticks_xformatted : list,         ticks_yformatted : list,
                    ticks_x          : list  = None, ticks_y          : list  = None,
                    range_xlim       : tuple = None, range_ylim       : tuple = None):
        """ Formats the graph axis
        This function formats the axis with the given values 
        
        Inputs: 
            label_x          - the x axis label
            label_y          - the y axis label 
            ticks_x          - where the x ticks are placed in the graph with respect 
                                    to the x range values
            ticks_y          - where the y ticks are placed in the graph with respect 
                                    to the y range values
            ticks_xformatted - what the x ticks are labelled with, left to right
            ticks_yformatted - what the y ticks are labelled with, top to bottom (reversed)
            range_xlim       - x bounds of the graph
            range_ylim       - y bounds of the graph

        """

        self.ax.set_xlabel("\n".join(wrap(label_x,60))) # warps if the axis label has more than 60 characters.
        self.ax.set_ylabel("\n".join(wrap(label_y,60))) # Do not want characters to be cut off in the graph

        # for heatmaps these are already intialized, because they are the same for all graphs
        if range_xlim is not None:
            self.ax.set_xlim(range_xlim)

        if range_ylim is not None: 
            self.ax.set_ylim(range_ylim)

        if ticks_x is not None:
            self.ax.set_xticks(ticks_x)

        if ticks_y is not None:
            self.ax.set_yticks(ticks_y)

        self.ax.set_xticklabels(ticks_xformatted, fontsize = 5, ha='right', rotation=45)
        self.ax.set_yticklabels(ticks_yformatted, fontsize = 5, ha='right')

    def get_axis_metadata(self, 
                          minimum : int,
                          maximum : int,
                          padding : int = 5):
        """ Provides values for defining the limits the axis
        
        Inputs: 
            minimum - the minimum of the data being plotted
            maximum - the maximum of the data being plotted
            padding - percentage of padding around the edges of data
        Returns:
            series_lim           - a tuple with the graph limits 
            series_tickplacement - values for defining the location of the ticks
        """
        if minimum == maximum:
            minimum -= 1
            maximum += 1
        
        series_range = maximum - minimum

        if padding > 0:
            seriesrange_padding = series_range/(100/padding) # how much do you want to pad?
        else:
            seriesrange_padding = 0
    
        series_minlim = minimum - seriesrange_padding
        series_maxlim = maximum + seriesrange_padding

        series_lim = (series_minlim, series_maxlim)
        series_tickplacement = np.linspace(series_minlim, series_maxlim, self.nticks)

        return series_lim, series_tickplacement


    def plot_histogram(self, series : pd.core.series.Series, 
                             output_filename : str):
        """ Plot Histograms
        
        Inputs:
            series      - data to plot
            output_file - location of where the graph gets saved
        """

        logger.debug(f"Plotting histogram for {series.name}")
        series_range, series_tickvals = self.get_axis_metadata(series.min(), series.max(), padding=5) # need to use current series max and series min 
        
        self.ax.set_xlim(series_range) # 5 percent padding because 100/20 = 5 
        self.ax.set_xticks(series_tickvals)
        

        if self.scale == "log":
            tick_valsformatted = format_ticks(self.convert_tolog(series_tickvals))
        else:
            tick_valsformatted = format_ticks(series_tickvals)
        
        self.ax.set_xticklabels(tick_valsformatted, fontsize = 5, ha='right', rotation=45)

        self.ax.hist(series, bins=self.bins, color=self.color)

        series_frequencyticks = self.ax.get_yticks()
        self.ax.set_yticklabels(format_ticks(series_frequencyticks), fontsize = 5, ha='right')

        self.fig.suptitle(series.name)
        self.ax.patch.set_facecolor(self.color)
        self.ax.patch.set_alpha(0.05)

        # this is faster than self.fig.savefig()
        self.fig.canvas.draw()
        output_plot = np.array(self.fig.canvas.renderer.buffer_rgba())
        imageio.imwrite(output_filename, output_plot, format='PNG-FI',compression=1)

        if os.path.exists(output_filename):
            self.ax.cla()


class HeatMap(Figures):

    """ This class Generates Heat Maps """

    def __init__(self, bins, CHUNK_SIZE, color, scale, stats, output_dir, nrows=None):
        Figures.__init__(self, bins, CHUNK_SIZE, color, scale, stats, output_dir)

        self.range_lims, self.range_tickvals = self.get_axis_metadata(0, self.bins, padding = 0)
        self.range_tickvals_offset = self.range_tickvals - 0.5 # the ticks are in the center of bins,
        # want the ticks on the edge of the bin

        self.clear_axis()
        self.dtype = get_dtype(nrows)

    def clear_axis(self):
        self.ax.cla()
        self.ax.grid(linestyle='dotted')
        self.ax.set_xticks(list(self.range_tickvals_offset))
        self.ax.set_yticks(list(self.range_tickvals_offset))

    def save_heatmap(self, 
                     output_image : np.ndarray,
                     output_filename : str,
                     seriesx_name : str, seriesy_name : str,
                     ticks_xformatted : list, ticks_yformatted : list,
                     seriesx_zero : int, seriesy_zero : int):
        """ This function formats the axis and saves the figure 

        Inputs:
            output_image     - numpy array containing the binned data
            output_filename  - location of where the graph gets saved
            seriesx_name     - name of column used to plot the x data
            seriesy_name     - name of oclumn used to plot the y data
            ticks_xformatted - name of x tick labels, left to right
            ticks_yformatted - name of y tick labels, top to bottom
            seriesx_zero     - binned value of zero on x axis 
            seriesy_zero     - binned value of zero on y axis
        """

        self.format_axis(label_x          = seriesx_name,     label_y          = seriesy_name,
                         ticks_xformatted = ticks_xformatted, ticks_yformatted = reversed(ticks_yformatted))

        datacolor = self.ax.pcolorfast(output_image,cmap=get_cmap())
        self.ax.imshow(np.rot90(output_image), cmap=get_cmap())

        if (self.stats['min'][seriesx_name] < 0):
            self.ax.axvline(x=seriesx_zero-1.5, linewidth=1) # add 1.5 for the offset

        if (self.stats['min'][seriesy_name] < 0):
            self.ax.axhline(y=self.bins-seriesy_zero+0.5, linewidth=1)
        
        # this is faster than self.fig.savefig()
        self.fig.canvas.draw()
        output_plot = np.array(self.fig.canvas.renderer.buffer_rgba())
        imageio.imwrite(output_filename, output_plot,format='PNG-FI',compression=1)

        logger.debug(f"Saved to: {output_filename}")
        self.clear_axis()

    def plot_graph(self,
                   series : pd.core.frame.DataFrame, 
                   output_filename_1 : str, 
                   output_filename_2 : str):
        """ This function gets individuals data for plotting heatmaps.
        Inputs:
            series            - dataframe containing x and y datapoints
            output_filename_1 - location of one of the two graphs saved 
            output_filename_2 - location of one of the two graphs saved
        """

        # do not want to plot data that is null
        series_ind = pd.notnull(series)
        series[~series_ind] = 255

        # separate out the series
        series1 = series.iloc[:,0]
        series2 = series.iloc[:,1]

        logger.debug(f"\n Graphing: {series1.name}, {series2.name}")
        # only need to update the labels of the graphs
        ticks_1 = (np.linspace(self.stats['bin_min'][series1.name], self.stats['bin_max'][series1.name], self.nticks))
        ticks_2 = (np.linspace(self.stats['bin_min'][series2.name], self.stats['bin_max'][series2.name], self.nticks))
        logger.debug(f"Tick 1 Values : {ticks_1}")
        logger.debug(f"Tick 2 Values : {ticks_2}")
        
        # for plotting x=0 and y=0
        series1_zero = abs(self.stats['bin_min'][series1.name]//self.stats['binwidth'][series1.name])
        series2_zero = abs(self.stats['bin_min'][series2.name]//self.stats['binwidth'][series2.name])

        if self.scale == "log": #would be in linear
            ticks_1 = self.convert_tolog(ticks_1)
            ticks_2 = self.convert_tolog(ticks_2)

        ticks_1formatted = format_ticks(ticks_1) #make it pretty
        ticks_2formatted = format_ticks(ticks_2)
        logger.debug(f"Tick 1 Labels : {ticks_1formatted}")
        logger.debug(f"Tick 2 Labels : {ticks_2formatted}")

        # get data to build the heatmaps
        sorted_feats = np.sort((series1 * self.bins) + series2)
        index        = np.nonzero(np.diff(sorted_feats))[0]
        index        = np.append(index,sorted_feats.size-1)
        rows         = (sorted_feats[index]/self.bins).astype(np.uint8) 
        cols         = np.mod(sorted_feats[index],self.bins) 
        counts       = np.diff(index)

        """ LOOP UNROLL ONCE: Plot the First Graph """
        # plot the data into a numpy array
        output_image_1 = np.zeros((self.bins, self.bins), dtype=self.dtype)
        output_image_1[rows[0], cols[0]] = index[0] + 1
        output_image_1[rows[1:],cols[1:]] = counts

        self.save_heatmap(output_image     = output_image_1,
                          output_filename  = output_filename_1,
                          seriesx_zero     = series1_zero,     seriesy_zero     = series2_zero,
                          seriesx_name     = series1.name,     seriesy_name     = series2.name,
                          ticks_xformatted = ticks_1formatted, ticks_yformatted = ticks_2formatted)

        """ LOOP UNROLL ONCE: Plot the Second Graph """
        # plot the data into a numpy array
        output_image_2 = np.zeros((self.bins, self.bins), dtype=self.dtype)
        output_image_2[cols[0], rows[0]] = index[0] + 1
        output_image_2[cols[1:],rows[1:]] = counts

        self.save_heatmap(output_image     = output_image_2,
                          output_filename  = output_filename_2,
                          seriesx_zero     = series2_zero,     seriesy_zero     = series1_zero,
                          seriesx_name     = series2.name,     seriesy_name     = series1.name,
                          ticks_xformatted = ticks_2formatted, ticks_yformatted = ticks_1formatted)


class ScatterPlot(Figures):

    """ This class Generates Heat Maps """

    def __init__(self, bins, CHUNK_SIZE, color, scale, stats, output_dir, nrows=None):
        Figures.__init__(self, bins, CHUNK_SIZE, color, scale, stats, output_dir)

    def plot_graph(self, 
                   series : pd.core.frame.DataFrame,
                   output_filename_1 : str, 
                   output_filename_2 : str):

        """ This function gets individuals data for scatterplots.
        Inputs:
            series            - dataframe containing x and y datapoints
            output_filename_1 - location of one of the two graphs saved 
            output_filename_2 - location of one of the two graphs saved
        """

        # do not want to plot data that is null
        series_ind = pd.notnull(series)
        series[~series_ind] = 255

        # separate out the series
        series1 = series.iloc[:,0]
        series2 = series.iloc[:,1]

        # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
        xy = np.vstack([series1, series2])
        z  = gaussian_kde(xy)(xy)

        logger.debug(f"\n Graphing: {series1.name}, {series2.name}")
        range_1_lims, range_1_tickvals = self.get_axis_metadata(minimum=series1.min(), maximum=series1.max(), padding=5)
        range_2_lims, range_2_tickvals = self.get_axis_metadata(minimum=series2.min(), maximum=series2.max(), padding=5)
        logger.debug(f"Range 1 Lims and Tick Placement: {range_1_lims} \n\t{range_1_tickvals}")
        logger.debug(f"Range 2 Lims and Tick Placement: {range_2_lims} \n\t{range_2_tickvals}")

        if self.scale == "log":
            range_1_formattedticks = format_ticks(self.convert_tolog(range_1_tickvals))
            range_2_formattedticks = format_ticks(self.convert_tolog(range_2_tickvals))
        else:
            # cannot override
            range_1_formattedticks = format_ticks(range_1_tickvals) # still need range_1_tickvals to format the axis
            range_2_formattedticks = format_ticks(range_2_tickvals) # still need range_2_tickvals to format the axis
        logger.debug(f"Range 1 Tick Labels: \n{range_1_formattedticks} \n{range_1_formattedticks}")
        logger.debug(f"Range 2 Tick Labels: \n{range_2_formattedticks} \n{range_2_formattedticks}")

        plt.tight_layout()

        """ LOOP UNROLL ONCE: Plot the First Graph """
        self.format_axis(label_x          = series1.name,            label_y          = series2.name,
                         ticks_x          = range_1_tickvals,        ticks_y          = range_2_tickvals,
                         ticks_xformatted = range_1_formattedticks,  ticks_yformatted = range_2_formattedticks,
                         range_xlim       = range_1_lims,            range_ylim       = range_2_lims)

        self.ax.scatter(series1, series2, c=z, s=100)
        # this is faster than self.fig.savefig()
        self.fig.canvas.draw()
        output_plot = np.array(self.fig.canvas.renderer.buffer_rgba())
        imageio.imwrite(output_filename_1, output_plot, format='PNG-FI',compression=1)

        logger.debug(f"Saved to: {output_filename_1}")
        self.ax.cla()

        """ LOOP UNROLL ONCE: Plot the Second Graph """
        self.format_axis(label_x          = series2.name,            label_y          = series1.name,
                         ticks_x          = range_2_tickvals,        ticks_y          = range_1_tickvals,
                         ticks_xformatted = range_2_formattedticks,  ticks_yformatted = range_1_formattedticks,
                         range_xlim       = range_2_lims,            range_ylim       = range_1_lims)

        self.ax.scatter(series2, series1, c=z, s=100)
        # this is faster than self.fig.savefig()
        self.fig.canvas.draw()
        output_plot = np.array(self.fig.canvas.renderer.buffer_rgba())
        imageio.imwrite(output_filename_2, output_plot, format='PNG-FI',compression=1)

        logger.debug(f"Saved to: {output_filename_2}")
        self.ax.cla()

