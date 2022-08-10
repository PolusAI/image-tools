import os, logging, argparse, json
import filepattern
from pathlib import Path
from utils import TextCell, TextLayerSpec, OverlaySpec, to_bijective


# Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def main(
    stitching_vector,
    outDir,
    file_pattern=None,
    concatenate=False
    ):


    def get_var_stats(vp):

        var_order = []
        var_stats = {v:{} for v in vp.variables}
        
        # Get the top left grid (0,0) to determine the top left image
        for file in vp():
            if file[0]['gridX'] == '0' and file[0]['gridY'] == '0':
                for v in vp.variables:
                    var_stats[v]['top_left'] = file[0][v]
                break
        
        for v in vp.variables:

            files = [file for file in vp(group_by=v)]
            
            # Check if the variable is constant over the stitching vector if so
            # then it is a top grid level variable
            if len(files[0]) < 2:
                x_grid = [int(file[0]['gridX']) for file in files]
                y_grid = [int(file[0]['gridY']) for file in files]
                
                x_grid.sort()
                y_grid.sort()
                
                x_range = x_step = x_grid[-1] - x_grid[0]
                y_range = y_step = y_grid[-1] - y_grid[0]
                
                var_stats[v]['x_range'] = x_range
                var_stats[v]['y_range'] = y_range
                
                var_stats[v]['x_step'] = x_step
                var_stats[v]['y_step'] = y_step
                
            else:
                # Get the grid range for each variable which represents its
                # total range across its grid minus the size of its child grid
                x_grid = [int(file['gridX']) for file in files[0]]
                y_grid = [int(file['gridY']) for file in files[0]]
                
                x_grid.sort()
                y_grid.sort()
            
                x_range = x_grid[-1] - x_grid[0]
                y_range = y_grid[-1] - y_grid[0]
                var_stats[v]['x_range'] = x_range
                var_stats[v]['y_range'] = y_range
            
                # Get the grid step for each variable; the step represents the 
                # distance to the next image when only the variable being 
                # considered is allowed to vary.
                x_step = x_grid[1] - x_grid[0]
                y_step = y_grid[1] - y_grid[0]
                var_stats[v]['x_step'] = x_step
                var_stats[v]['y_step'] = y_step
                
            
            # Save the variable, step size and x,y or w variable type. A x 
            # variable means the variable only vaires in the x dimension and y 
            # variable only varies in the y dimension. A w variable means it 
            # varies in both dimensions
            if y_range == 0:
                var_order.append([x_step, v, 'x', x_range])
                var_stats[v]['v_type'] = 'x'
                
            elif x_range == 0:
                var_order.append([y_step, v, 'y', y_range])
                var_stats[v]['v_type'] = 'y'
                
            else:
                step = max(x_step, y_step)
                var_order.append([step, v, 'w', x_range, y_range])
                var_stats[v]['v_type'] = 'w'
        
        # Sort from largest to smallest step size
        var_order.sort(reverse=True)
        
        logger.debug(var_stats)
        logger.debug(var_order)
        
        layout = ''
    
        # Define the current sub-grid length and height
        x_factor = 1
        y_factor = 1
        
        grid_dim = []
        
        while var_order:
            
            # Define the current variable name, step and type
            v_data = var_order.pop()
            v = v_data[1]
            t = v_data[2]
            
            # Check if variable is a multi-dimensional variable 
            # (w stands for wrap)
            if t == 'w':
                
                x_range = var_stats[v]['x_range']
                y_range = var_stats[v]['y_range']
                
                if var_stats[v]['x_step'] > var_stats[v]['y_step']:
                    s = v_data[0]/x_factor
                    
                else:
                    s = v_data[0]/x_factor
                
                if var_order:
                    layout += v + ','
                    
                else:
                    layout += v
            
            elif t == 'x':
                
                s = v_data[0]/x_factor
                
                if var_order:
                    nv_data = var_order[-1]
                    nv = nv_data[1]
                    ns = nv_data[0]/y_factor
                    nt = nv_data[2]
                    
                    # Try to match the x variable with a y variable
                    if s == ns and nt == 'y':
                        layout += v + nv + ','
                        var_order.pop()
                        
                        x_range = var_stats[v]['x_range']
                        y_range = var_stats[nv]['y_range']
                    
                    else:
                        layout += v + ','
                        x_range = var_stats[v]['x_range']
                        y_range = 0

                else:
                    layout += v  
                    x_range = var_stats[v]['x_range']
                    y_range = 0
                        
                        
            elif t == 'y':
                
                s = v_data[0]/y_factor
                
                if var_order:
                    nv_data = var_order[-1]
                    nv = nv_data[1]
                    ns = nv_data[0]/x_factor
                    nt = nv_data[2]
                    
                    # Try to match the y variable with a x variable
                    if s == ns and var_stats[nv]['v_type'] == 'x':
                        layout += nv + v + ','
                        var_order.pop()
                        
                        x_range = var_stats[nv]['x_range']
                        y_range = var_stats[v]['y_range']

                    else:
                        layout += v + ','
                        y_range = var_stats[v]['y_range']
                        x_range = 0

                else:
                    layout += v
                    y_range = var_stats[v]['y_range']
                    x_range = 0
                    
                    
            # The factors represent the number of images in the current sub-grid
            # for thier respective axis
            x_factor += x_range
            y_factor += y_range
            
            grid_dim.append((x_factor, y_factor))

        return var_stats, layout


    def generate_overlays(layout, grid, vp, var_stats):
        
        overlays = {level:[] for level in range(len(layout))}
        variables = {level+1:{} for level in range(len(layout)-1)}
        levels = len(layout)
        
        
        # Define the variable and parent variable for each grid level
        for i, v in enumerate(layout[:-1]):
                
            level = i + 1
            variables[level]['v'] = v
            
            if len(v) == 2:
                variables[level]['vx'] = v[0]
                variables[level]['vy'] = v[1]
                
            # Get the parent grid variables
            pv = layout[i+1]
            variables[level]['pv'] = pv
            
            if len(pv) == 2:
                variables[level]['px'] = pv[0]
                variables[level]['py'] = pv[1]
        
        # Iterate over every file in the vector pattern
        for file in grid:
            
            text = file['file']
            position = (file['posX'], file['posY'])
            
            # Add the base level (single image overlay)
            overlays[0].append(
                TextCell(position=position, text=text)
                )
            
            # For each grid level check if the image is a corner image
            for i in range(levels-1):
                
                level = i + 1
                
                # Define the current level's variables and parent variables
                v = variables[level]['v']
                pv = variables[level]['pv']
                    
                if len(v) == 2:
                    vx = variables[level]['vx']
                    vy = variables[level]['vy']
                    fx = file[vx]
                    fy = file[vy]

                    # Check if the image is top left for current grid level
                    if fx == var_stats[vx]['top_left'] and fy == var_stats[vy]['top_left']:
                        
                        # Get the parent grid data
                        if len(pv) == 2:
                            px = variables[level]['px']
                            py = variables[level]['py']
                            
                            # This ensures a range of 0 to x where x is the
                            # number of images in a row
                            fpx = abs(file[px] - var_stats[px]['top_left'])
                            
                            # This ensures a range of A to y where y is the
                            # number of images in a column (in bijective num.)
                            fpy = abs(file[py] - var_stats[py]['top_left']) + 1
                            
                            # Use bijective numeration for dual variable levels 
                            text = to_bijective(int(fpy)) + str(fpx)
                            overlays[level].append(
                                TextCell(position=position, text=text)
                                )
                        
                        else:
                            text = pv.upper() + str(file[pv])
                            overlays[level].append(
                                TextCell(position=position, text=text)
                                )
                        
                    else:
                        # If the sub-grid is not a corner image then it cannot 
                        # be a corner image for parent grids
                        break
                
                else:
                    fp = file[v]
                    
                    if fp == var_stats[v]['top_left']:
                        
                        # Get the parent grid data
                        if len(pv) == 2:
                            px = variables[level]['px']
                            py = variables[level]['py']
                            
                            # This ensures a range of 0 to x where x is the
                            # number of images in a row
                            fpx = abs(file[px] - var_stats[px]['top_left'])
                            
                            # This ensures a range of A to y where y is the
                            # number of images in a column (in bijective num.)
                            fpy = abs(file[py] - var_stats[py]['top_left']) + 1
                            
                            # Use bijective numeration for dual variable levels 
                            text = to_bijective(int(fpy)) + str(fpx)
                            overlays[level].append(
                                TextCell(position=position, text=text)
                                )
                        
                        else:
                            text = pv.upper() + str(file[pv])
                            overlays[level].append(
                                TextCell(position=position, text=text)
                                )
                        
                    else:
                        # If the sub-grid is not a corner image then it cannot 
                        # be a corner image for parent grids
                        break
        
        return overlays
    
    
    stitching_vector_paths = []
    
    # Define the stitching vector path
    for path in Path(stitching_vector).iterdir():
        if 'img-global-positions' in path.name:
            stitching_vector_paths.append(path)
    
    if file_pattern is None:
        logger.info("Infering filepattern...")
        
        files = []
        
        # Parse the image file names in the stiching vector
        with stitching_vector_paths[0].open('r') as fhand:
            for line in fhand.readlines():
                for component in line.split("; "):
                    elements = component.split(": ")
                    if elements[0] == "file":
                        file = elements[1]
                        files.append(file)
                        break
        
        vector_pattern = filepattern.infer_pattern(files)
        logger.info("Inferred filepattern = {}".format(vector_pattern))

    text_overlays = []
    
    for vector_path in stitching_vector_paths:
        vp = filepattern.VectorPattern(
            file_path=vector_path, 
            pattern=file_pattern
            )
        
        var_stats, layout = get_var_stats(vp)
        grid = [f[0] for f in vp()]
        layout = layout.split(',')

        overlays_by_level = generate_overlays(layout, grid, vp, var_stats)
        overlays = [text_cell for level in overlays_by_level.values() 
                    for text_cell in level]
        
        output_name = vp.output_name()
        output_path = outDir.joinpath(output_name)
        
        tls = TextLayerSpec(
            id=output_name,
            width=0,
            height=0,
            cell_size=0,
            data=overlays
            )
        
        if concatenate:
            text_overlays.append(output_path)

        overlay = OverlaySpec(text_layers=[tls])
        with output_path.open("w") as fw:
            json.dump(
                overlay.dict(by_alias=True),
                fw,
                indent=2,
            )
        
    if concatenate:
        
        text_overlays.sort()
        combined_output_path = outDir.joinpath('overlay.json')
        num_paths = len(text_overlays)
        logger.info('Combining {} text overlay files in {}'.format(num_paths, combined_output_path))
        
        # Create the overlay fine with single open array bracket
        with open(combined_output_path, 'w') as fw:
            fw.write('[')
            fw.close()
        
        # Combine all the overlay files
        for i, path in enumerate(text_overlays):
            
            # Load the overlay data
            with path.open('r') as fr:
                data = json.load(fr)
                fr.close()
        
            # Write the overlay data
            with combined_output_path.open('a') as fa:
                
                fa.write(json.dumps(data, indent=2))
                
                if i != num_paths - 1:
                    fa.write(',\n')
                    
                else:
                    fa.write(']')
                    
                fa.close()

if __name__ == '__main__':

    # Setup Command Line Arguments
    logger.info("Parsing arguments...")

    # Instantiate argparser object
    parser = argparse.ArgumentParser(
        prog="main", description="Polus Render Overlay Plugin"
    )

    # Add the plugin arguments
    parser.add_argument(
        "--stitchingVector", 
        dest="stitchingVector", 
        type=str, 
        help="Path to the stitching vector directory", 
        required=True
    )
    
    parser.add_argument(
        "--outDir", 
        dest="outDir", 
        type=str, 
        help="Path to the stitching vector directory", 
        required=True
    )    
    
    parser.add_argument(
        "--filePattern", 
        dest="filePattern", 
        type=str, 
        help="The stitching vector file pattern", 
        required=False
        )
    
    parser.add_argument(
        "--concatenate", 
        dest="concatenate", 
        type=bool, 
        help="If the output be a single file", 
        required=False
        )


    # Parse and log the arguments
    args = parser.parse_args()
    
    stitching_vector = Path(args.stitchingVector)
    logger.info('stitchingVector = {}'.format(stitching_vector))
    
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))

    file_pattern = args.filePattern
    logger.info('filePattern = {}'.format(file_pattern))
    
    concatenate = args.concatenate
    logger.info('concatenate = {}'.format(concatenate))

    main(
        stitching_vector=stitching_vector,
        outDir=outDir,
        file_pattern=file_pattern,
        concatenate=concatenate
        )
