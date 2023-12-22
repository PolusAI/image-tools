
import logging
from pathlib import Path
from bfio import BioReader, BioWriter
import filepattern as fp
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# length/width of the chunk each _merge_layers thread processes at once
chunk_size = 8192

THREADS = ThreadPoolExecutor()._max_workers

# Units for conversion
UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1}


def _z_stacking(inp_dir:Path, file_pattern:str,  out_dir:Path):
        
        zpositions= []
        input_files = []

        fps = fp.FilePattern(inp_dir, file_pattern)
        out_name = fps.output_name()
        for file in fps(group_by="z"):
            f1, f2 = file
            if f1[0][0] == "z":
                zpos = f1[0][1]
                zfile = f2[0][1][0]
                input_files.append(zfile)
                zpositions.append(zpos)

        # Get the number of layers to stack
        z_size = len(zpositions)
        
        # Get some basic info about the files to stack
        with BioReader(input_files[0]) as br:

            # Get the physical z-distance if available, set to physical x if not
            ps_z = br.ps_z
            
            # If the z-distances are undefined, average the x and y together
            if None in ps_z:
                # Get the size and units for x and y
                x_val,xunits = br.ps_x
                y_val,yunits = br.ps_y

                x_units = xunits.value
                y_units = yunits.value
                
                # Convert x and y values to the same units and average
                z_val = (x_val*UNITS[x_units] + y_val*UNITS[y_units])/2
                
                # Set z units to the smaller of the units between x and y
                z_units = x_units if UNITS[x_units] < UNITS[y_units] else y_units
                
                # Convert z to the proper unit scale
                z_val /= UNITS[z_units]
                ps_z = (z_val,z_units)

                if not ps_z:
                    raise ValueError(f"Could not find physical z-size {ps_z}. Using the average of x and y")
              

            # Hold a reference to the metadata once the file gets closed
            metadata = br.metadata

        # Create the output file within a context manager
        with BioWriter(out_dir.joinpath(out_name),metadata=metadata,max_workers=THREADS) as bw:

            # Adjust the dimensions before writing
            bw.Z = z_size
            bw.ps_z = ps_z

            # ZIndex tracking for the output file
            zi = 0

            # Start stacking
            for file in input_files:

                # Open an image
                with BioReader(file,max_workers=THREADS) as br:

                    # Open z-layers one at a time
                    for z in range(br.Z):

                        # Use tiled reading in x&y to conserve memory
                        # At most, [chunk_size, chunk_size] pixels are loaded
                        for y in range(0,br.Y,chunk_size):
                            y_max = min([br.Y,y + chunk_size])
                            for x in range(0,br.X,chunk_size):
                                x_max = min([br.X,x + chunk_size])

                                bw[y:y_max,x:x_max,zi:zi+1,...] = br[x:x_max,y:y_max,z:z+1,...]

                        zi += 1

def _t_stacking(inp_dir:Path, file_pattern:str,  out_dir:Path):
        
        timepoints= []
        input_files = []

        fps = fp.FilePattern(inp_dir, file_pattern)
        out_name = fps.output_name()
        for file in fps(group_by="t"):
            f1, f2 = file
            if f1[0][0] == "t":
                zpos = f1[0][1]
                zfile = f2[0][1][0]
                input_files.append(zfile)
                timepoints.append(zpos)

        # Get the number of layers to stack
        t_size = len(timepoints)

        with BioReader(input_files[0]) as br:
            metadata = br.metadata


        with BioWriter(out_dir.joinpath(out_name),metadata=metadata,max_workers=THREADS) as bw:
  
            # Adjust the dimensions before writing
            bw.T = t_size
    
            # ZIndex tracking for the output file
            ti = 0

            for file in input_files:

                # Open an image
                with BioReader(file,max_workers=THREADS) as br:

                    for t in range(br.T):

                        for y in range(0,br.Y,chunk_size):
                            y_max = min([br.Y, y + chunk_size])

                            for x in range(0,br.X,chunk_size):
                                x_max = min([br.X, x + chunk_size])

                                tile = br[y:y_max,x:x_max, 0, 0, t:t+1]

                                bw[y:y_max,x:x_max,0,0 ,ti:ti+1] = tile

                        ti += 1

def _channel_stacking(inp_dir:Path, file_pattern:str,  out_dir:Path):
        
        channels= []
        input_files = []

        fps = fp.FilePattern(inp_dir, file_pattern)
        out_name = fps.output_name()
        for file in fps(group_by="c"):
            f1, f2 = file
            if f1[0][0] == "c":
                zpos = f1[0][1]
                zfile = f2[0][1][0]
                input_files.append(zfile)
                channels.append(zpos)

        # Get the number of layers to stack
        ch_size = len(channels)
        
        with BioReader(input_files[0]) as br:
            metadata = br.metadata


        with BioWriter(out_dir.joinpath(out_name),metadata=metadata,max_workers=THREADS) as bw:
  
            # Adjust the dimensions before writing
            bw.C = ch_size
    
            # ZIndex tracking for the output file
            ci = 0

            for file in input_files:

                # Open an image
                with BioReader(file,max_workers=THREADS) as br:

                    for c in range(br.C):

                        for y in range(0,br.Y,chunk_size):
                            y_max = min([br.Y, y + chunk_size])

                            for x in range(0,br.X,chunk_size):
                                x_max = min([br.X, x + chunk_size])

                                tile = br[y:y_max,x:x_max, 0, c:c+1, 0]

                                bw[y:y_max,x:x_max,0,ci:ci+1 ,0] = tile

                        ci += 1


def _dimension_stacking(inp_dir:Path, file_pattern:str,  group_by:list[str], out_dir:Path):
        
        # channels= []
        # input_files = []

        fps = fp.FilePattern(inp_dir, file_pattern)
        out_name = fps.output_name()
        for file in fps(group_by=group_by):
            print(file)
        #     f1, f2 = file
        #     if f1[0][0] == "c":
        #         zpos = f1[0][1]
        #         zfile = f2[0][1][0]
        #         input_files.append(zfile)
        #         channels.append(zpos)

        # # Get the number of layers to stack
        # ch_size = len(channels)
        
        # with BioReader(input_files[0]) as br:
        #     metadata = br.metadata


        # with BioWriter(out_dir.joinpath(out_name),metadata=metadata,max_workers=THREADS) as bw:
  
        #     # Adjust the dimensions before writing
        #     bw.C = ch_size
    
        #     # ZIndex tracking for the output file
        #     ci = 0

        #     for file in input_files:

        #         # Open an image
        #         with BioReader(file,max_workers=THREADS) as br:

        #             for c in range(br.C):

        #                 for y in range(0,br.Y,chunk_size):
        #                     y_max = min([br.Y, y + chunk_size])

        #                     for x in range(0,br.X,chunk_size):
        #                         x_max = min([br.X, x + chunk_size])

        #                         tile = br[y:y_max,x:x_max, 0, c:c+1, 0]

        #                         bw[y:y_max,x:x_max,0,ci:ci+1 ,0] = tile

        #                 ci += 1