import argparse, logging

import os
from matplotlib.pyplot import ticklabel_format
import numpy as np
import pandas as pd

from Dataset import LinearData, LogData
from Pyramid import GraphPyramid
from Figure import Figures
from Figure import HeatMap
from Figure import ScatterPlot

from concurrent.futures import ProcessPoolExecutor

POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG', 'INFO'))
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

CHUNK_SIZE = 1024

def plots_2D(graphPyramid,
             bincount, 
             dataset,
             color,
             output_path,
             graphing,
             tick_format,
             combo_names):
    

    if graphing == "heatmap": # bin the data
        figures = HeatMap(bins=bincount, 
                          CHUNK_SIZE=CHUNK_SIZE, 
                          color=color, scale=dataset.scale, 
                          stats=dataset.stats,
                          tick_format=tick_format,
                          output_dir=output_path,
                          nrows = dataset.nexamples)
    
    else: # plotting every single data point
        figures = ScatterPlot(bins=bincount, 
                                CHUNK_SIZE=CHUNK_SIZE, 
                                color=color, scale=dataset.scale, 
                                stats=dataset.stats,
                                tick_format=tick_format,
                                output_dir=output_path)

    logger.info("Generating 2D plots ...")
    for data_x, data_y in combo_names:
        # we create both graphs on either side of the diagonal; only need to iterate through the combinations once
        
        logger.info(f"Generating plots for {data_x} and {data_y}")
        figures.plot_graph(series            = dataset.dataframe[[data_x, data_y]],
                           output_filename_1 = os.path.join(graphPyramid.bottom_pyramidDir, 
                                                                graphPyramid.base_imagenames[(data_x, data_y)]),
                           output_filename_2 = os.path.join(graphPyramid.bottom_pyramidDir, 
                                                                graphPyramid.base_imagenames[(data_y, data_x)]))

def histograms(graphPyramid,
               bincount, 
               dataset,
               color,
               tick_format,
               output_path,
               column_names,
               graphing=None):
    
    figures = Figures(bins=bincount, 
                      color=color, 
                      scale=dataset.scale, 
                      stats=dataset.stats, 
                      CHUNK_SIZE=CHUNK_SIZE,
                      tick_format=tick_format,
                      output_dir=output_path)

    logger.info("Generating 1D plots ...")
    for data in column_names:

        logger.info(f"Generating histograms for {data}")
        figures.plot_histogram(dataset.dataframe[data], 
                               os.path.join(graphPyramid.bottom_pyramidDir, graphPyramid.base_imagenames[(data,data)]))

    

def main(input_path  : str,
         output_path : str,
         scales      : str,
         graphing    : str,
         bincount    : int,
         tick_format : str,
         color       : str):

    if scales == "linear" or scales == "log":
        scales = [scales]
    else:
        scales = ["linear", "log"]

    input_files = os.listdir(input_path)

    for input_basecsv in input_files:
        for scale in scales:
            input_csvpath : str = os.path.join(input_path, input_basecsv)
            file_basename : str = input_basecsv[:-4] + f"_{scale}_{graphing}"


            if scale == "linear":
                dataset = LinearData(input_csvpath)
            if scale == "log":
                dataset = LogData(input_csvpath) # using symlog 

            # need to initailze the graph Pyramid to get the appropraite directories 
            graphPyramid = GraphPyramid(output_dir=output_path, output_name=file_basename, \
                                        ngraphs=dataset.ngraphs, axisnames=dataset.plot_combinations, \
                                        CHUNK_SIZE=CHUNK_SIZE)

            num_cpus = os.cpu_count()

            # the graphs need to be done in groups when using multi processing, because each core 
                # initializes its own figure to fill out for the graphs
            data_kwargs = {
                'graphPyramid' : graphPyramid,
                'bincount'     : bincount,
                'dataset'      : dataset,
                'color'        : color,
                'tick_format'  : tick_format,
                'output_path'  : output_path,
                'graphing'     : graphing,
            }

            with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
                num_columns = len(dataset.column_names)
                processor_count_1D = num_columns//num_cpus + 1
                logger.debug(f"All {num_columns} columns are Split into {num_cpus} groups, " +
                            f"so that each processor (group) can run {processor_count_1D} Columns " +
                            "to build histograms")
                for i in range(0, num_columns, processor_count_1D):
                    list_columns = list(dataset.column_names[i:i+processor_count_1D])
                    i += processor_count_1D
                    executor.submit(histograms, 
                                    column_names = list_columns,
                                    **data_kwargs)

            # binning only for heatmaps
            if graphing == "heatmap":
                logger.info("Binning the Data ...")
                dataset.bin_data(bincount=bincount)
                data_kwargs['dataset'] = dataset # need to update kwargs

            with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
                num_combos = len(dataset.plot_combinations_unique)
                processor_count_2D = num_combos//num_cpus + 1
                logger.debug(f"All {num_columns} columns are Split into {num_cpus} groups, " +
                             f"so that each processor (group) can run {processor_count_1D} Columns " +
                               "to build {graphing} (plots/maps)}")

                for i in range(0, num_combos, processor_count_2D):
                    list_columns = list(dataset.plot_combinations_unique[i:i+processor_count_2D])
                    i += processor_count_2D
                    executor.submit(plots_2D,
                                    combo_names = list_columns,
                                    **data_kwargs)

            # need to use the images in the base directory to build rest of the pyramid for deepzooming
            logger.info("Building up the Pyramid ...")
            graphPyramid.build_thepyramid()


if __name__=="__main__":
    
    
    """ Initialize argument parser """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Build an image pyramid from data in a csv file.')

    """ Define the arguments """
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Path to input images.', required=True)

    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Path to output images.', required=True)
    
    parser.add_argument('--graphing', dest='graphing', type=str,
                        help='What types of graphs are you making?', required=True)

    parser.add_argument('--bincount', dest='bin_count', type=int,
                        help='Number of bins for histograms and heatmaps', required=True)
    
    parser.add_argument('--scale', dest='scale', type=str,
                        help='Linear, Log, or Both', required=True)

    parser.add_argument('--color', dest='color', type=str,
                        help='Pick your favorite color!', required=False, default='blue')
    
    parser.add_argument('--tickFormat', dest='tick_format', type=str,
                        help="What kind of tick labels do you want?", required=False, default='scientific_notation')

    """ Get the input arguments """
    args = parser.parse_args()

    input_path : str  = args.inpDir
    output_path : str = args.outDir
    assert os.path.exists(output_path)

    graphing : str = args.graphing

    bincount : int    = args.bin_count
    if bincount:
        assert bincount > 10, "Bincount Must be Greater than 10"

    scales : str = args.scale.lower()
    assert ((scales == "linear") or (scales == "log") or scales == "both"), \
        f"Check your Scale Parameter, you have it defined as {scales}, but only linear, log, or both are acceptable"

    color : str = args.color
    tick_format : str = args.tick_format

    logger.info(f'inpDir      = {input_path}')
    logger.info(f'outDir      = {output_path}')
    logger.info(f'binCount    = {bincount}')
    logger.info(f'scaling     = {scales}')
    logger.info(f'color       = {color}')
    logger.info(f'tick_format = {tick_format}')

    main(input_path  = input_path,
         output_path = output_path,
         bincount    = bincount,
         graphing    = graphing,
         scales      = scales,
         tick_format = tick_format,
         color       = color)