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
logger.setLevel("DEBUG")

# define separately for importing for unit testing
convert_tolog = lambda t : np.sign(t)*(1/np.log(10))*(-1+(10**abs(t)))

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
        self.format_ticks  = lambda tick: np.format_float_scientific(tick, precision=3, min_digits=3)

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
        range_xlims, ticks_x = self.get_axis_metadata(series.min(), series.max(), padding=5) # need to use current series max and series min 
        
        self.ax.set_xlim(range_xlims) # 5 percent padding because 100/20 = 5 
        self.ax.set_xticks(ticks_x)
        

        if self.scale == "log":
            ticks_xformatted = list(map(self.format_ticks, self.convert_tolog(ticks_x)))
        else:
            ticks_xformatted = list(map(self.format_ticks, ticks_x))
        
        self.ax.set_xticklabels(ticks_xformatted, fontsize = 5, ha='right', rotation=45)

        self.ax.hist(series, bins=self.bins, color=self.color)

        ticks_y = self.ax.get_yticks()
        ticks_yformatted = list(map(self.format_ticks, ticks_y))
        self.ax.set_yticklabels(ticks_yformatted, fontsize = 5, ha='right')

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
        series_ind = pd.notnull(series) # method in first Version of Plugin
        series[~series_ind] = 255

        # separate out the series
        series1 = series.iloc[:,0]
        series2 = series.iloc[:,1]

        logger.debug(f"\n Graphing: {series1.name}, {series2.name}")
        
        # for plotting x=0 and y=0 axis
        series1_zero = abs(self.stats['bin_min'][series1.name]//self.stats['binwidth'][series1.name])
        series2_zero = abs(self.stats['bin_min'][series2.name]//self.stats['binwidth'][series2.name])

        # only need to update the labels of the graphs
        ticks_1 = (np.linspace(self.stats['bin_min'][series1.name], self.stats['bin_max'][series1.name], self.nticks))
        ticks_2 = (np.linspace(self.stats['bin_min'][series2.name], self.stats['bin_max'][series2.name], self.nticks))
        
        if self.scale == "log": #would be in linear
            ticks_1 = self.convert_tolog(ticks_1)
            ticks_2 = self.convert_tolog(ticks_2)
        logger.debug(f"Tick 1 Values (Placement):\n{ticks_1}")
        logger.debug(f"Tick 2 Values (Placement):\n{ticks_2}")

        ticks_1formatted = list(map(self.format_ticks, ticks_1)) #make it pretty
        ticks_2formatted = list(map(self.format_ticks, ticks_2))
        logger.debug(f"Tick 1 Labels:\n{ticks_1formatted}")
        logger.debug(f"Tick 2 Labels:\n{ticks_2formatted}")

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
        series.dropna()

        # separate out the series
        series1 = series.iloc[:,0]
        series2 = series.iloc[:,1]

        logger.debug(f"\n Graphing: {series1.name}, {series2.name}")
        plot_density = True # can sometimes have bad data
        try:
            # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
            xy = np.vstack([series1, series2])
            z  = gaussian_kde(xy)(xy)
        except:
            plot_density = False

        range_1lims, ticks_1 = self.get_axis_metadata(minimum=series1.min(), maximum=series1.max(), padding=5)
        range_2lims, ticks_2 = self.get_axis_metadata(minimum=series2.min(), maximum=series2.max(), padding=5)
        logger.debug(f"Range 1 Lims: {range_1lims}")
        logger.debug(f"Range 2 Lims: {range_2lims}")
        logger.debug(f"Tick 1 Values (Placement):\n{ticks_1}")
        logger.debug(f"Tick 2 Values (Placement):\n{ticks_2}")

        if self.scale == "log":
            ticks_1formatted = list(map(self.format_ticks, self.convert_tolog(ticks_1)))
            ticks_2formatted = list(map(self.format_ticks, self.convert_tolog(ticks_2)))
        else:
            # cannot override
            ticks_1formatted = list(map(self.format_ticks, ticks_1)) # still need ticks_1 as a separate variable to format the axis
            ticks_2formatted = list(map(self.format_ticks, ticks_2)) # still need ticks_2 as a separate variable to format the axis
        logger.debug(f"Ticks 1 Labels:\n{ticks_1formatted}")
        logger.debug(f"Ticks 2 Labels:\n{ticks_2formatted}")

        plt.tight_layout()

        """ LOOP UNROLL ONCE: Plot the First Graph """
        self.format_axis(label_x          = series1.name,     label_y          = series2.name,
                         ticks_x          = ticks_1,          ticks_y          = ticks_2,
                         ticks_xformatted = ticks_1formatted, ticks_yformatted = ticks_2formatted,
                         range_xlim       = range_1lims,      range_ylim       = range_2lims)

        if plot_density:
            self.ax.scatter(series1, series2, c=z, s=100)
        else:
            self.ax.scatter(series1, series2, color=self.color)
    
        # this is faster than self.fig.savefig()
        self.fig.canvas.draw()
        output_plot = np.array(self.fig.canvas.renderer.buffer_rgba())
        imageio.imwrite(output_filename_1, output_plot, format='PNG-FI',compression=1)

        logger.debug(f"Saved to: {output_filename_1}")
        self.ax.cla()

        """ LOOP UNROLL ONCE: Plot the Second Graph """
        self.format_axis(label_x          = series2.name,     label_y          = series1.name,
                         ticks_x          = ticks_2,          ticks_y          = ticks_1,
                         ticks_xformatted = ticks_2formatted, ticks_yformatted = ticks_1formatted,
                         range_xlim       = range_2lims,      range_ylim       = range_1lims)

        if plot_density:
            self.ax.scatter(series2, series1, c=z, s=100)
        else:
            self.ax.scatter(series2, series1, color=self.color)

        # this is faster than self.fig.savefig()
        self.fig.canvas.draw()
        output_plot = np.array(self.fig.canvas.renderer.buffer_rgba())
        imageio.imwrite(output_filename_2, output_plot, format='PNG-FI',compression=1)

        logger.debug(f"Saved to: {output_filename_2}")
        self.ax.cla()

